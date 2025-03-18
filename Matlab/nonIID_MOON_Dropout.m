clc;
clear;
close all;
delete(gcp("nocreate"));
%% Define Dataset Path
DatasetPath = fullfile('Dataset_nonIID'); 
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

    % add new conv
    convolution2dLayer([5 1], 16, 'Name', 'conv3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool3')

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
CommunicationRounds = 100; 
LocalEpochs = 10; 
LearningRate = 0.001;
Momentum = 0.5;
Velocity = []; 
Temperature = 0.5;
Mu = 1.0;
dropout_round = 10;
drop_client_ids = [1, 2, 5, 6];


% server published a global model to all participants
localModel = globalModel;
%% Define Monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info="CommunicationRound", ...
    XLabel="Communication Round");
%% Training Circuit
Round = 0;
spmd
    PreLocRepresent = []; 
end

%  record the accuracy of each class for global test
GlobalRecording = zeros(CommunicationRounds, NumClasses);
GlobalAccuracyRecord = zeros(1, CommunicationRounds);

% stop conditions
while Round < CommunicationRounds && ~Monitor.Stop 

    Round = Round + 1;

    spmd
        % Simulated client drop: If the current round >=dropout_round and the current client is the specified dropped client, local training is skipped
        if (Round >= dropout_round) && ismember(spmdIndex, drop_client_ids)
            % Client offline, directly synchronize the global model, skip the calculation
            localModel.Learnables.Value = globalModel.Learnables.Value;
            locLearnable = localModel.Learnables.Value;
        else
            % Normal local training process
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
            locLearnable = localModel.Learnables.Value;
        end
    end

    % Set the sample number of dropped clients to 0 during global aggregation so that it does not affect global updates
    locTrainSizeCell = cell(1, participants);
    for k = 1:participants
        if ismember(k, drop_client_ids)
            locTrainSizeCell{k} = 0;
        else
            locTrainSizeCell{k} = locTrainSize{k};
        end
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
save('GlobalTestAccuracyRecordforDropOut.mat', 'GlobalAccuracyRecord');
save('GlobalClassTestAccuracyRecordforDropOut.mat', 'GlobalRecording');