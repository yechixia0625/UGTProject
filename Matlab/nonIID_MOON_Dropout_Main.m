clc;
clearvars;
close all;
delete(gcp("nocreate"));
%% Define Dataset Path
DatasetPath = fullfile('Dataset_nonIID_simplex'); 
%% Define Parallel
cluster = parcluster("Processes");
cluster.NumWorkers = 6;
parpool = parpool(cluster); 
%% Define Participant
participants = parpool.NumWorkers;
%% Define Image Augmentation
inputSize = [160 20 1]; 
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
gloTest = augmentedImageDatastore(inputSize(1:2),gloTest);
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
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu4')

    % block 4
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu5')

    % block 5
    fullyConnectedLayer(NumClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')];
  
globalModel = dlnetwork(layers);
%% Define Global Constants
CommunicationRounds = 50; 
LocalEpochs = 10; 
LearningRate = 0.0001;
Momentum = 0.5;
Velocity = []; 
Temperature = 0.8;
Mu = 1.0;
dropout_round = 20;
drop_client_ids = [6];

% server published a global model to all participants
localModel = globalModel;
%% Define Monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info="CommunicationRound", ...
    XLabel="Communication Round");
%% Training Circuit
Round = 0;

% Initialize the preLocalModel parameter for each client
PreLocalModelLearnables = cell(1, participants);
for i = 1:participants
    PreLocalModelLearnables{i} = globalModel.Learnables.Value;
end

%  record the accuracy of each class for global test
GlobalRecording = zeros(CommunicationRounds, NumClasses);
GlobalAccuracyRecord = zeros(1, CommunicationRounds);

% stop conditions
while Round < CommunicationRounds && ~Monitor.Stop 

    Round = Round + 1;

    spmd
        if (Round >= dropout_round) && ismember(spmdIndex, drop_client_ids)
            % For dropout clients after the dropout round, simply sync with the global model
            localModel.Learnables.Value = globalModel.Learnables.Value;
            locLearnable = localModel.Learnables.Value;
        else
            % Non-dropout clients: initialize with the global model and perform local training
            preLocalModelParams = PreLocalModelLearnables{spmdIndex};
            preLocalModel = localModel;
            preLocalModel.Learnables.Value = preLocalModelParams;
            localModel.Learnables.Value = globalModel.Learnables.Value;
            for epoch = 1:LocalEpochs
                shuffle(locTrainMBQ);
                while hasdata(locTrainMBQ)
                    [X, Y] = next(locTrainMBQ);
                    [loss, gradient] = dlfeval(@FedMOONLossGrad, localModel, globalModel, X, Y, preLocalModel, Temperature, Mu);
                    [localModel, Velocity] = sgdmupdate(localModel, gradient, Velocity, LearningRate, Momentum);
                end
            end
            newPreLocalModelParams = localModel.Learnables.Value;
            locLearnable = localModel.Learnables.Value;
        end
    end

    % Set the sample number of dropped clients to 0 during global aggregation so that it does not affect global updates
    locTrainSizeCell = cell(1, participants);
    for k = 1:participants
        if ismember(k, drop_client_ids) && Round >= dropout_round
            locTrainSizeCell{k} = 0;
        else
            locTrainSizeCell{k} = locTrainSize{k};
        end
    end
    
    % Update the preLocalModel parameter for each client
    for i = 1:participants
        PreLocalModelLearnables{i} = newPreLocalModelParams{i};
    end
    locFactor = [locTrainSizeCell{:}] / sum([locTrainSizeCell{:}]);
    globalModel.Learnables.Value = FederatedAveraging(locFactor, locLearnable);

    %% Global Model Evaluation 
    [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);
    fprintf('Round %d: Global Test Accuracy: %.4f\n', Round, GlobalTestAccuracy);
    
    % Records test accuracy and various accuracy rates
    GlobalAccuracyRecord(Round) = GlobalTestAccuracy;
    GlobalRecording(Round, :) = GlobalClassAccuracy;
    
    % Draw the confusion matrix
    plotconfusion(GlobalTestLabel, GlobalTestPred);
    
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);     
    Monitor.Progress = 100 * Round / CommunicationRounds;
end

FinalRoundEachClassAccuracy = GlobalRecording(Round, :);
if ~isempty(drop_client_ids)
    dropClientStr = "_drop_client_ids_" + strjoin(string(drop_client_ids), '_');
else
    dropClientStr = "";
end

%% Save the result and the ploting
if ~exist('MOON_dropout_result', 'dir')
    mkdir('MOON_dropout_result');
end

filenameAccuracy = fullfile('MOON_dropout_result', sprintf('GlobalTestAccuracyRecordforSimplex%s.mat', char(dropClientStr)));
save(filenameAccuracy, 'GlobalAccuracyRecord');

filenameClass = fullfile('MOON_dropout_result', sprintf('GlobalClassTestAccuracyRecordforSimplex%s.mat', char(dropClientStr)));
save(filenameClass, 'GlobalRecording');

fig2 = figure('Visible','off');
epochs_global = 1:CommunicationRounds;
plot(epochs_global, GlobalAccuracyRecord, '-', 'LineWidth', 1.5);
xlabel('Epoch');
ylabel('Global Test Accuracy');
title('Global Test Accuracy vs Epoch');
grid on;
filename2 = fullfile('MOON_dropout_result', sprintf('GlobalTestAccuracy%s.png', char(dropClientStr)));
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
filename3 = fullfile('MOON_dropout_result', sprintf('PerClassTestAccuracy%s.png', char(dropClientStr)));
saveas(fig3, filename3);
close(fig3);