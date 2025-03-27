%% Clear workspace
clc;
clearvars -except drop_client_ids;
close all;
delete(gcp("nocreate"));

%% Define the data set path
DatasetPath = fullfile('Dataset_nonIID'); 

%% Set up parallel computing
cluster = parcluster("Processes");
cluster.NumWorkers = 6;
parpool = parpool(cluster); 
participants = parpool.NumWorkers;

%% Define Image Augmentation
inputSize = [160 20 1]; 

%% Print the dropout client ID
fprintf('drop_client_ids: %d\n', drop_client_ids);

%% Each Participant Use 70% For Training and 10% For Validation
spmd   
    DatasetPath = fullfile(DatasetPath, ['local_', num2str(spmdIndex)]);
    locSet = imageDatastore(DatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % local training set 
    [locTrain, locTest] = splitEachLabel(locSet, 0.8, "randomized");
    % local validation set
    [locTrain, locVal] = splitEachLabel(locTrain, 0.875, "randomized");

    locTrain = augmentedImageDatastore(inputSize(1:2), locTrain);
    locVal = augmentedImageDatastore(inputSize(1:2), locVal);
end

%% Server Use Each 20% For Testing
XList = [];
YList = [];

% Collect file paths and labels from each client's test set
for i = 1:participants
    template = locTest{i};
    XList = [XList; template.Files];
    YList = [YList; template.Labels];       
end
gloTest = imageDatastore(XList, 'Labels', YList);

% Get class categories and number of classes
classes = categories(gloTest.Labels);
NumClasses = numel(classes);

% Apply image augmentation (resize images) to the global test set
gloTest = augmentedImageDatastore(inputSize(1:2), gloTest);

%% Dataset Preprocessing
MiniBatchSize = 100;

% Define preprocessing function
preprocess = @(X,Y)MiniBatchPreprocessing(X,Y,classes); 

spmd
    % Record the number of observations in the local training set
    locTrainSize = locTrain.NumObservations;
    
    % Create mini-batch queue for the local training set
    locTrainMBQ = minibatchqueue(locTrain, ...
        MiniBatchSize = MiniBatchSize, ...
        MiniBatchFcn = preprocess, ...
        MiniBatchFormat = ["SSCB",""]);

    % Create mini-batch queue for the local validation set
    locValMBQ = minibatchqueue(locVal, ...
        MiniBatchSize = MiniBatchSize, ...
        MiniBatchFcn = preprocess, ...
        MiniBatchFormat = ["SSCB",""]); 
end

% Create mini-batch queue for the global test set
gloTsetMBQ = minibatchqueue(gloTest, ...
    MiniBatchSize = MiniBatchSize, ...
    MiniBatchFcn = preprocess, ...
    MiniBatchFormat = ["SSCB",""]);  

%% Define the Network
layers = [
    imageInputLayer(inputSize, Normalization = "none")
    
    % block 1
    convolution2dLayer([5 1], 16, 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool1')
    
    % block 2
    convolution2dLayer([5 1], 16, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool2')
    
    % block 3
    convolution2dLayer([5 1], 16, 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool3')
    
    % block 4
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')
    
    % block 5
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu5')
    
    % block 6
    fullyConnectedLayer(NumClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')];
  
globalModel = dlnetwork(layers);

%% Gets the index and initial values of the fc3 layer parameters in the global model
idxW = find(strcmp(globalModel.Learnables.Layer, 'fc3') & strcmp(globalModel.Learnables.Parameter, 'Weights'));
idxB = find(strcmp(globalModel.Learnables.Layer, 'fc3') & strcmp(globalModel.Learnables.Parameter, 'Bias'));
W_fc3 = globalModel.Learnables.Value{idxW};
b_fc3 = globalModel.Learnables.Value{idxB};
globalSimplexLR = 0.001;

%% Define Global Constants
CommunicationRounds = 50; 
simplex_start_epoch = 11;
dropout_round = 20;
LocalEpochs = 10;
LearningRate = 0.001;
Momentum = 0.5;
Velocity = []; 
Temperature = 0.5;
Mu = 1.0;

% The server publishes the global model to each client
localModel = globalModel;

%% Initializes the record variable
numClients = participants; 
GlobalRecording = zeros(CommunicationRounds, NumClasses);
GlobalAccuracyRecord = zeros(1, CommunicationRounds);
AlphaHistory = zeros(CommunicationRounds, participants);
alphaOld = ones(1, participants) / participants;

%% Initialize previous round fc3 parameters for dropout compensation
prev_global_W = W_fc3;
prev_global_b = b_fc3;
prev_alpha = alphaOld;
prev_clientW_fc3 = cell(1, numClients);
prev_clientb_fc3 = cell(1, numClients);
for i = 1:numClients
    prev_clientW_fc3{i} = W_fc3;
    prev_clientb_fc3{i} = b_fc3;
end

% Initialize dropout compensation weights
dropoutCompensationDone = false;
dropout_clientW_comp = cell(1, numClients);
dropout_clientb_comp = cell(1, numClients);

% Initialize fc3 history
clientW_history = cell(CommunicationRounds, 1);
clientb_history = cell(CommunicationRounds, 1);

%% Define the training progress monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info = "CommunicationRound", ...
    XLabel = "Communication Round");

%% Initialize internal variables for spmd
Round = 0;
spmd
    PreLocRepresent = [];
end

%% Federated Learning
while Round < CommunicationRounds && ~Monitor.Stop
    Round = Round + 1;
    
    %% Local Training Update (per client)
    spmd
        if (Round >= dropout_round) && ismember(spmdIndex, drop_client_ids)
            % For dropout clients after the dropout round, simply sync with the global model
            localModel.Learnables.Value = globalModel.Learnables.Value;
            locLearnable = localModel.Learnables;
        else
            % Non-dropout clients: initialize with the global model and perform local training
            localModel.Learnables.Value = globalModel.Learnables.Value;
            for epoch = 1:LocalEpochs
                shuffle(locTrainMBQ);
                while hasdata(locTrainMBQ)
                    [X, Y] = next(locTrainMBQ);
                    [CurLocalRepresent, gradient] = dlfeval(@FedMOONLossGrad, localModel, globalModel, X, Y, PreLocRepresent, Temperature, Mu);
                    [localModel, Velocity] = sgdmupdate(localModel, gradient, Velocity, LearningRate, Momentum);
                    PreLocRepresent = CurLocalRepresent;
                end
            end
            locLearnable = localModel.Learnables;
        end
    end
    
    %% Collect fc3 layer gradients from all clients
    spmd
        if ~hasdata(locTrainMBQ)
            reset(locTrainMBQ);
        end
        [X_dummy, Y_dummy] = next(locTrainMBQ);
        [dummyLoss, dummyGradients] = dlfeval(@FedMOONLossGrad, localModel, globalModel, X_dummy, Y_dummy, PreLocRepresent, Temperature, Mu);
        % Extract gradients for fc3 layer weights and biases
        fc3W_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Weights')};
        fc3B_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Bias')};
        gradVector = [fc3W_grad(:); fc3B_grad(:)];
        localGradientVector = gradVector;
    end
    gradAll = localGradientVector;
    
    %% Compute the latest aggregation weights (alpha) using the new gradients
    if Round < simplex_start_epoch
        alpha = alphaOld; 
    else
        newAlpha_grad = computeAlphaFromGradients(gradAll, numClients);
        if Round < dropout_round
            % Before dropout round: use the simplex learning result directly
            currentAlpha = newAlpha_grad;
        else
            % During dropout rounds: for dropout clients, mix the predicted value with newAlpha_grad
            currentAlpha = newAlpha_grad;
            for i = 1:numClients
                if ismember(i, drop_client_ids)
                    xdata = (1:(Round-1))';
                    ydata = AlphaHistory(1:(Round-1), i);
                    f = fit(xdata, ydata, 'poly1');
                    predicted_value = f(Round);
                    blending_factor = 0.3;
                    currentAlpha(i) = blending_factor * predicted_value + (1 - blending_factor) * newAlpha_grad(i);
                end
            end
        end
        currentAlpha = currentAlpha / sum(currentAlpha);
        alpha = currentAlpha;
    end
    AlphaHistory(Round, :) = alpha;
    alphaOld = alpha;
    fprintf('The alpha value of round %d is: %s\n', Round, num2str(alpha));
    
    %% Update fc3 layer parameters using the latest alpha
    if Round < dropout_round
        % Before dropout round: perform standard gradient aggregation update on fc3 layer
        allGradientsMat = [];
        for i = 1:numClients
            grad_i = extractdata(gradAll{i});
            allGradientsMat = [allGradientsMat, grad_i];
        end
        globalGrad = zeros(size(allGradientsMat,1), 1);
        for k = 1:numClients
            globalGrad = globalGrad + alpha(k) * allGradientsMat(:, k);
        end
        numW = numel(W_fc3);
        gradW = globalGrad(1:numW);
        gradB = globalGrad(numW+1:end);
        gradW = reshape(gradW, size(W_fc3));
        gradB = reshape(gradB, size(b_fc3));
        
        % Update fc3 weights and biases
        W_fc3 = W_fc3 - globalSimplexLR * gradW;
        b_fc3 = b_fc3 - globalSimplexLR * gradB;
        
        spmd
            clientW = localModel.Learnables.Value{idxW};
            clientb = localModel.Learnables.Value{idxB};
        end
        clientW_cell = cell(1, numClients);
        clientb_cell = cell(1, numClients);
        for i = 1:numClients
            clientW_cell{i} = clientW{i};
            clientb_cell{i} = clientb{i};
        end
    else
        % During dropout rounds: use a compensation to update dropout clients' fc3 layer
        spmd
            clientW = localModel.Learnables.Value{idxW};
            clientb = localModel.Learnables.Value{idxB};
        end
        clientW_cell = cell(1, numClients);
        clientb_cell = cell(1, numClients);
        for i = 1:numClients
            clientW_cell{i} = clientW{i};
            clientb_cell{i} = clientb{i};
        end
        
        if ~dropoutCompensationDone
            [W_fc3, b_fc3, clientW_cell, clientb_cell] = compensateDropoutWeights(...
                alpha, clientW_cell, clientb_cell, drop_client_ids, ...
                prev_global_W, prev_global_b, prev_alpha, prev_clientW_fc3, prev_clientb_fc3, ...
                AlphaHistory, clientW_history, clientb_history, Round);
            dropout_clientW_comp = clientW_cell;
            dropout_clientb_comp = clientb_cell;
            dropoutCompensationDone = true;
        else
            for i = 1:numClients
                if ismember(i, drop_client_ids)
                    clientW_cell{i} = dropout_clientW_comp{i};
                    clientb_cell{i} = dropout_clientb_comp{i};
                end
            end
            new_global_W = zeros(size(prev_global_W));
            new_global_b = zeros(size(prev_global_b));
            for i = 1:numClients
                new_global_W = new_global_W + alpha(i) * clientW_cell{i};
                new_global_b = new_global_b + alpha(i) * clientb_cell{i};
            end
            W_fc3 = new_global_W;
            b_fc3 = new_global_b;
        end
    end
    
    %% Global Aggregation (FedAvg): Update other layers (excluding fc3)
    spmd
        locLearnable = localModel.Learnables;
    end
    locTrainSizeCell = cell(1, numClients);
    for k = 1:numClients
        % Exclude dropout clients from FedAvg aggregation
        if ismember(k, drop_client_ids)
            locTrainSizeCell{k} = 0;
        else
            locTrainSizeCell{k} = locTrainSize{k};
        end
    end
    locFactor = [locTrainSizeCell{:}] / sum([locTrainSizeCell{:}]);
    globalLearnable = FederatedAveragingforSimplexLearning(locFactor, locLearnable);
    
    % Replace fc3 layer parameters in the global model with the updated ones
    globalLearnable.Value{idxW} = W_fc3;
    globalLearnable.Value{idxB} = b_fc3;
    globalModel.Learnables = globalLearnable;
        
    %% Test global model performance
    [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);
    fprintf('Round %d: Global Test Accuracy: %.4f\n', Round, GlobalTestAccuracy);
    GlobalAccuracyRecord(Round) = GlobalTestAccuracy;
    GlobalRecording(Round, :) = GlobalClassAccuracy;
    
    plotconfusion(GlobalTestLabel, GlobalTestPred);
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);
    Monitor.Progress = 100 * Round / CommunicationRounds;
    
    %% Update previous round parameters for dropout compensation
    prev_global_W = W_fc3;
    prev_global_b = b_fc3;
    prev_alpha = alpha;
    prev_clientW_fc3 = clientW_cell;
    prev_clientb_fc3 = clientb_cell;
    
    clientW_history{Round} = clientW_cell;
    clientb_history{Round} = clientb_cell;
end

FinalRoundEachClassAccuracy = GlobalRecording(Round, :);

if ~isempty(drop_client_ids)
    dropClientStr = "_drop_client_ids_" + strjoin(string(drop_client_ids), '_');
else
    dropClientStr = "";
end

%% Save the result and the ploting
if ~exist('Simplex_result', 'dir')
    mkdir('Simplex_result');
end

filenameAccuracy = fullfile('Simplex_result', sprintf('GlobalTestAccuracyRecordforSimplex%s.mat', char(dropClientStr)));
save(filenameAccuracy, 'GlobalAccuracyRecord');

filenameClass = fullfile('Simplex_result', sprintf('GlobalClassTestAccuracyRecordforSimplex%s.mat', char(dropClientStr)));
save(filenameClass, 'GlobalRecording');

filenameAlpha = fullfile('Simplex_result', sprintf('AlphaHistory%s.mat', char(dropClientStr)));
save(filenameAlpha, 'AlphaHistory');

figAlpha = figure('Visible','off');
for i = 1:participants
    subplot(3,2,i);
    plot(1:CommunicationRounds, AlphaHistory(:, i), '-');
    xlabel('Communication Round');
    ylabel('Alpha Value');
    ylim([0 1]);
    title(sprintf('Client %d', i));
    grid on;
end
sgtitle('Alpha Evolution per Client');
filenameAlpha = fullfile('Simplex_result', sprintf('AlphaEvolutionTrend%s.png', char(dropClientStr)));
saveas(figAlpha, filenameAlpha);
close(figAlpha);

fig2 = figure('Visible','off');
epochs_global = 1:CommunicationRounds;
plot(epochs_global, GlobalAccuracyRecord, '-', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Global Test Accuracy');
title('Global Test Accuracy vs Epoch');
grid on;
filename2 = fullfile('Simplex_result', sprintf('GlobalTestAccuracy%s.png', char(dropClientStr)));
saveas(fig2, filename2);
close(fig2);

fig3 = figure('Visible','off');
epochs_global = 1:CommunicationRounds;
for c = 1:NumClasses
    subplot(3,2,c);
    plot(epochs_global, GlobalRecording(:, c), '-', 'LineWidth', 1.5);
    xlabel('Epoch');
    ylabel('Test Accuracy');
    title(sprintf('Class %d', c));
    grid on;
end
sgtitle('Per-Class Test Accuracy vs Epoch');
ax3 = findall(gcf, 'Type', 'axes');
linkaxes(ax3, 'y');
filename3 = fullfile('Simplex_result', sprintf('PerClassTestAccuracy%s.png', char(dropClientStr)));
saveas(fig3, filename3);
close(fig3);