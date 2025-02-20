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
    reluLayer('Name', 'relu3')

    % block 4
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu3')

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
        % Synchronize global model to local
        localModel.Learnables.Value = globalModel.Learnables.Value; 
        
        % Local Epochs
        for epoch = 1:LocalEpochs
            shuffle(locTrainMBQ); 
            while hasdata(locTrainMBQ) 
                [X, Y] = next(locTrainMBQ); 
                [CurLocalRepresent, gradient] = dlfeval(@FedMOONLossGrad, ...
                    localModel, globalModel, X, Y, PreLocRepresent, Temperature, Mu);
                [localModel, Velocity] = sgdmupdate(localModel, gradient, ...
                    Velocity, LearningRate, Momentum);
                PreLocRepresent = CurLocalRepresent;
            end
        end
        % Collect local model parameters after local training
        locLearnable = localModel.Learnables.Value;
    end

    % ============ Simplex Learning ============

    % Gather parameter differences for all layers
    numClients = participants;
    numLayers = height(globalModel.Learnables);
    allGradients = [];  

    for k = 1:numClients
        paramDiff_k = [];
        for row = 1:numLayers
            globalParam = globalModel.Learnables.Value{row};
            localParam  = locLearnable{k}{row};
            
            diffRow = localParam - globalParam;
            paramDiff_k = [paramDiff_k; diffRow(:)];
        end
        allGradients = [allGradients, extractdata(paramDiff_k)];
    end

    % PCA dimension = numClients - 1 (for the simplex)
    simplexDim = numClients - 1;
    [coeff, ~, ~] = pca(allGradients');
    topPC = coeff(:, 1:simplexDim);
    reducedGradients = topPC' * allGradients; 

    % Apply Riesz s-Energy regularization
    s = 0.1;
    lambda_riesz = 0.01;
    regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda_riesz);

    % Project onto simplex
    simplexPoints = zeros(numClients, simplexDim);
    for k = 1:numClients
        simplexPoints(k,:) = project_to_simplex(regularizedGradients(:,k));
    end

    % Compute pairwise distances, get similarity, then sample alpha
    distMatrix = pdist2(simplexPoints, simplexPoints);
    simMatrix = 1 ./ (1 + distMatrix);
    simMatrix = simMatrix ./ sum(simMatrix, 2);

    alphaCount = zeros(1, numClients);
    for k = 1:numClients
        chosen = randsample(1:numClients, 1, true, simMatrix(k,:));
        alphaCount(chosen) = alphaCount(chosen) + 1;
    end
    alpha = alphaCount / sum(alphaCount);

    % Update the entire global model using alpha
    for row = 1:numLayers
        newParam = 0;
        for k = 1:numClients
            localParam = locLearnable{k}{row};
            newParam = newParam + alpha(k) * localParam;
        end
        globalModel.Learnables.Value{row} = newParam;
    end
    
    % ============ End of Simplex Learning aggregator ============


    %% Global Model Evaluation 
    [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = ...
        EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);

    fprintf('Round %d: Global Test Accuracy: %.4f\n', Round, GlobalTestAccuracy);

    % Record GlobalTestAccuracy
    GlobalAccuracyRecord(Round) = GlobalTestAccuracy;
    GlobalRecording(Round, :) = GlobalClassAccuracy;

    % plot confusion matrix
    plotconfusion(GlobalTestLabel, GlobalTestPred)

    % (Optional) save confusion matrix
    % outputDir = fullfile(pwd, 'confusionMATRIX');
    % if ~exist(outputDir, 'dir')
    %     mkdir(outputDir);
    % end
    % confusionFileName = fullfile(outputDir, ['ConfusionMatrix_Round_', num2str(Round), '.png']);
    % saveas(gcf, confusionFileName);
    % close(gcf);

    % Update monitor
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);     
    Monitor.Progress = 100 * Round / CommunicationRounds;
end

FinalRoundEachClassAccuracy = GlobalRecording(Round, :);
save('GlobalTestAccuracyRecordforSimplex.mat', 'GlobalAccuracyRecord');
save('GlobalClassTestAccuracyRecordforSimplex.mat', 'GlobalRecording');