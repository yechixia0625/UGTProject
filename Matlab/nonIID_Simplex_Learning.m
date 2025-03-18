%% 
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

% X: image, Y: label
for i = 1:participants
    template = locTest{i};
    XList = [XList; template.Files];
    YList = [YList; template.Labels];       
end
gloTest = imageDatastore(XList, 'Labels', YList);

% access classes
classes = categories(gloTest.Labels);
NumClasses = numel(classes);

% global test set
gloTest = augmentedImageDatastore(inputSize(1:2), gloTest);

%% Dataset Preprocessing
MiniBatchSize = 100;

% X: image, Y: label
preprocess = @(X,Y)MiniBatchPreprocessing(X,Y,classes); 

spmd
    locTrainSize = locTrain.NumObservations;
    
    locTrainMBQ = minibatchqueue(locTrain, ...
        MiniBatchSize = MiniBatchSize, ...
        MiniBatchFcn = preprocess, ...
        MiniBatchFormat = ["SSCB",""]);

    locValMBQ = minibatchqueue(locVal, ...
        MiniBatchSize = MiniBatchSize, ...
        MiniBatchFcn = preprocess, ...
        MiniBatchFormat = ["SSCB",""]); 
end

gloTsetMBQ = minibatchqueue(gloTest, ...
    MiniBatchSize = MiniBatchSize, ...
    MiniBatchFcn = preprocess, ...
    MiniBatchFormat = ["SSCB",""]);  

%% Define Network
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
simplex_start_epoch = 15;
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

%% Initializes the previous round of fc3 layers for subsequent dropout reconstruction
prev_global_W = W_fc3;
prev_global_b = b_fc3;
prev_alpha = alphaOld;
prev_clientW_fc3 = cell(1, numClients);
prev_clientb_fc3 = cell(1, numClients);
for i = 1:numClients
    prev_clientW_fc3{i} = W_fc3;
    prev_clientb_fc3{i} = b_fc3;
end

%% Define Monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info = "CommunicationRound", ...
    XLabel = "Communication Round");

%% Initializes spmd internal variables
Round = 0;
spmd
    PreLocRepresent = [];
end

%% Federated Learning
while Round < CommunicationRounds && ~Monitor.Stop
    Round = Round + 1;
    % For the dropout client, only the global model is synchronized
    spmd
        if (Round >= dropout_round) && ismember(spmdIndex, drop_client_ids)
            localModel.Learnables.Value = globalModel.Learnables.Value;
            locLearnable = localModel.Learnables;
        else
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
    
    % Collect the fc3 layer gradient of each client
    spmd
        if ~hasdata(locTrainMBQ)
            reset(locTrainMBQ);
        end
        [X_dummy, Y_dummy] = next(locTrainMBQ);
        [dummyLoss, dummyGradients] = dlfeval(@FedMOONLossGrad, localModel, globalModel, X_dummy, Y_dummy, PreLocRepresent, Temperature, Mu);
        fc3W_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Weights')};
        fc3B_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Bias')};
        gradVector = [fc3W_grad(:); fc3B_grad(:)];
        localGradientVector = gradVector;
    end
    gradAll = localGradientVector;
    
    if Round < dropout_round
        % Dropout round not reached: Update fc3 layer using normal gradient polymerization
        allGradientsMat = [];
        for i = 1:numClients
            grad_i = extractdata(gradAll{i});
            allGradientsMat = [allGradientsMat, grad_i];
        end
        globalGrad = zeros(size(allGradientsMat,1), 1);
        for k = 1:numClients
            globalGrad = globalGrad + alphaOld(k) * allGradientsMat(:, k);
        end
        numW = numel(W_fc3);
        gradW = globalGrad(1:numW);
        gradB = globalGrad(numW+1:end);
        gradW = reshape(gradW, size(W_fc3));
        gradB = reshape(gradB, size(b_fc3));
        
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
        
        [W_fc3, b_fc3, clientW_cell, clientb_cell] = compensateDropoutWeights(alphaOld, clientW_cell, clientb_cell, drop_client_ids, prev_global_W, prev_global_b, prev_alpha, prev_clientW_fc3, prev_clientb_fc3);
    end
    
    %% Update the fc3 layer parameters in the global model and use FedAvg aggregation for the other layers
    spmd
        locLearnable = localModel.Learnables;
    end

    % For the dropout client, its sample number is treated as 0 and does not participate in FedAvg weighting
    locTrainSizeCell = cell(1, numClients);
    for k = 1:numClients
        if ismember(k, drop_client_ids)
            locTrainSizeCell{k} = 0;
        else
            locTrainSizeCell{k} = locTrainSize{k};
        end
    end
    locFactor = [locTrainSizeCell{:}] / sum([locTrainSizeCell{:}]);
    globalLearnable = FederatedAveragingforSimplexLearning(locFactor, locLearnable);

    % Update global fc3 layer weights and bias
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
    
    %% Update the last round of parameters used for dropout reconstruction
    prev_global_W = W_fc3;
    prev_global_b = b_fc3;
    prev_alpha = alphaOld;
    prev_clientW_fc3 = clientW_cell;
    prev_clientb_fc3 = clientb_cell;
    
    %% Update alpha
    if Round < simplex_start_epoch
        alpha = alphaOld; 
    elseif Round == simplex_start_epoch
        newAlpha = computeAlphaFromGradients(gradAll, numClients);
        currentAlpha = newAlpha;
        alpha = currentAlpha;
    elseif mod(Round - simplex_start_epoch, 5) == 0
        newAlpha = computeAlphaFromGradients(gradAll, numClients);
        blending_factor = 0.5;
        currentAlpha = blending_factor * newAlpha + (1 - blending_factor) * currentAlpha;
        currentAlpha = currentAlpha / sum(currentAlpha);
        alpha = currentAlpha;
    else
        noise_scale = 0.001;
        noisy_alpha = currentAlpha + noise_scale * randn(size(currentAlpha));
        noisy_alpha(noisy_alpha < 0) = 0;
        if sum(noisy_alpha) == 0
            currentAlpha = ones(1, numClients) / numClients;
        else
            currentAlpha = noisy_alpha / sum(noisy_alpha);
        end
        alpha = currentAlpha;
    end
    AlphaHistory(Round, :) = alpha;
    alphaOld = alpha;
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

figAlpha = figure('Visible','off');
for i = 1:participants
    subplot(3,2,i);
    plot(1:CommunicationRounds, AlphaHistory(:, i), '-');
    xlabel('Communication Round');
    ylabel('Alpha Value');
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
