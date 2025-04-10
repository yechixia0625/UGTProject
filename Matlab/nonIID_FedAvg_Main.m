clc;
clear;
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

% server published a global model to all participants
localModel = globalModel;
%% Define Monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info="CommunicationRound", ...
    XLabel="Communication Round");
%% Training Circuit
Round = 0;
PreLocRepresent = [];

%  record the accuracy of each class for global test
GlobalRecording = zeros(CommunicationRounds, NumClasses);
GlobalAccuracyRecord = zeros(1, CommunicationRounds);

% stop conditions
while Round < CommunicationRounds && ~Monitor.Stop 

    Round = Round + 1;

    spmd
        % update local model
        localModel.Learnables.Value = globalModel.Learnables.Value; 
        % local epochs
        for epoch = 1:LocalEpochs
            % data shuffle
            shuffle(locTrainMBQ); 
            % mini-batch learning
            while hasdata(locTrainMBQ) 
                % X: image, Y: label
                [X, Y] = next(locTrainMBQ); 
                % loss and gradient calculation
                [loss, gradient] = dlfeval(@FedAvgLossGrad,localModel, X, Y); 
                % stochastic gradient descent update
                [localModel, Velocity] = sgdmupdate(localModel, gradient, Velocity, LearningRate, Momentum);
            end
        end
        % local model parameter collection
        locLearnable = localModel.Learnables.Value;
    end
    % federated averaging
    locFactor = [locTrainSize{:}] / sum([locTrainSize{:}]);
    globalModel.Learnables.Value = FederatedAveraging(locFactor, locLearnable);
    %% Global Model Evaluation 
    [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);
    fprintf('Round %d: Global Test Accuracy: %.4f\n', Round, GlobalTestAccuracy);
    % Record GlobalTestAccuracy
    GlobalAccuracyRecord(Round) = GlobalTestAccuracy;
    % record the accuracy of each class for global test
    GlobalRecording(Round, :) = GlobalClassAccuracy;
    % confusion matrix
    plotconfusion(GlobalTestLabel, GlobalTestPred)
    % update monitor
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);     
    Monitor.Progress = 100 * Round / CommunicationRounds;
end
FinalRoundEachClassAccuracy = GlobalRecording(Round, :);

%% Save the result and the ploting
if ~exist('FedAvg_result', 'dir')
    mkdir('FedAvg_result');
end

save('FedAvg_result/GlobalTestAccuracyRecordforFedAvgnonIID.mat', 'GlobalAccuracyRecord');
save('FedAvg_result/GlobalClassTestAccuracyRecordforFedAvgnonIID.mat', 'GlobalRecording');

load('FedAvg_result/GlobalTestAccuracyRecordforFedAvgnonIID.mat', 'GlobalAccuracyRecord');
load('FedAvg_result/GlobalClassTestAccuracyRecordforFedAvgnonIID.mat', 'GlobalRecording');

figure;
plot(GlobalAccuracyRecord, '-o','LineWidth', 2);
xlabel('Communication Rounds');
ylabel('Global Test Accuracy');
title('Global Test Accuracy over Communication Rounds');
grid on;
saveas(gcf, 'FedAvg_result/GlobalTestAccuracy.png');

figure;
numClasses = size(GlobalRecording,2);
rounds = 1:size(GlobalRecording,1);
hold on;
colors = lines(numClasses);
for c = 1:numClasses
    plot(rounds, GlobalRecording(:,c), '-o','LineWidth', 2, 'Color', colors(c,:));
end
xlabel('Communication Rounds');
ylabel('Test Accuracy per Class');
title('Global Test Accuracy for each Class');
legend(arrayfun(@(c) sprintf('Class %d', c), 1:numClasses, 'UniformOutput', false));
grid on;
hold off;
saveas(gcf, 'FedAvg_result/GlobalClassTestAccuracy.png');
