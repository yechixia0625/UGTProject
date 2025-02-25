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

dropout_round = 10;
drop_client_ids = [1, 2, 5, 6];

% server published a global model to all participants
localModel = globalModel;

idxW = find(strcmp(globalModel.Learnables.Layer, 'fc3') & strcmp(globalModel.Learnables.Parameter, 'Weights'));
idxB = find(strcmp(globalModel.Learnables.Layer, 'fc3') & strcmp(globalModel.Learnables.Parameter, 'Bias'));
W_fc3 = globalModel.Learnables.Value{idxW};
b_fc3 = globalModel.Learnables.Value{idxB};
globalSimplexLR = 0.001;

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
        if (Round >= dropout_round) && ismember(spmdIndex, drop_client_ids)
            localModel.Learnables.Value = globalModel.Learnables.Value;
            locLearnable = localModel.Learnables.Value;
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
            locLearnable = localModel.Learnables.Value;
        end
    end

    % Collect the fc3 layer gradient of each client
    spmd
        if ~hasdata(locTrainMBQ)
            reset(locTrainMBQ);
        end
        [X_dummy, Y_dummy] = next(locTrainMBQ);
        [dummyLoss, dummyGradients] = dlfeval(@fedavg_loss_gradient_calc, localModel, X_dummy, Y_dummy);
        fc3W_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Weights')};
        fc3B_grad = dummyGradients.Value{strcmp(dummyGradients.Layer, 'fc3') & strcmp(dummyGradients.Parameter, 'Bias')};
        gradVector = [fc3W_grad(:); fc3B_grad(:)];
        localGradientVector = gradVector;
    end

    gradAll = localGradientVector;  
    numClients = participants;
    allGradients = [];
    for i = 1:numClients
         grad_i = extractdata(gradAll{i});
         allGradients = [allGradients, grad_i];
    end

    simplexDim = numClients - 1;
    [coeff, ~, ~] = pca(allGradients');  
    reducedGradients = coeff(:, 1:simplexDim)';  

    s = 0.5;
    lambda = 0.01;
    regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda);

    simplexPoints = zeros(numClients, simplexDim);
    for k = 1:numClients
        simplexPoints(k,:) = project_to_simplex(regularizedGradients(:,k));
    end

    if Round > 5  
        similarityMatrix = pdist(simplexPoints);
        similarityMatrix = squareform(similarityMatrix);
        samplingWeights = 1 ./ (1 + similarityMatrix);
        samplingWeights = samplingWeights ./ sum(samplingWeights, 2);
        alpha = zeros(1, numClients);
        for i = 1:numClients
            alpha(i) = randsample(1:numClients, 1, true, samplingWeights(i,:));
        end
        alpha_weights = zeros(1, numClients);
        for i = 1:numClients
            alpha_weights(alpha(i)) = alpha_weights(alpha(i)) + 1;
        end
        alpha = alpha_weights ./ sum(alpha_weights);
    else 
        alpha = random_simplex_point(simplexDim);  
    end

    globalGrad = zeros(size(allGradients,1), 1);
    for k = 1:numClients
        globalGrad = globalGrad + alpha(k) * allGradients(:, k);
    end

    numW = numel(W_fc3);
    gradW = globalGrad(1:numW);
    gradB = globalGrad(numW+1:end);
    gradW = reshape(gradW, size(W_fc3));
    gradB = reshape(gradB, size(b_fc3));
    W_fc3 = W_fc3 - globalSimplexLR * gradW;
    b_fc3 = b_fc3 - globalSimplexLR * gradB;
    
    % FedAverage: Collects other layer parameters for each client
    spmd
        locLearnable = localModel.Learnables;
    end
    % On global aggregation, set the number of training samples to 0 for the dropout client
    locTrainSizeCell = cell(1, participants);
    for k = 1:participants
        if ismember(k, drop_client_ids)
            locTrainSizeCell{k} = 0;
        else
            locTrainSizeCell{k} = locTrainSize{k};
        end
    end
    locFactor = [locTrainSizeCell{:}] / sum([locTrainSizeCell{:}]);
    globalLearnable = FederatedAveragingforSimplexLearning(locFactor, locLearnable);
    globalLearnable.Value{idxW} = W_fc3;
    globalLearnable.Value{idxB} = b_fc3;
    globalModel.Learnables = globalLearnable;
    
    %% Global Model Evaluation 
    [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);
    fprintf('Round %d: Global Test Accuracy: %.4f\n', Round, GlobalTestAccuracy);
    GlobalAccuracyRecord(Round) = GlobalTestAccuracy;
    GlobalRecording(Round, :) = GlobalClassAccuracy;
    plotconfusion(GlobalTestLabel, GlobalTestPred);
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);     
    Monitor.Progress = 100 * Round / CommunicationRounds;
end

FinalRoundEachClassAccuracy = GlobalRecording(Round, :);
save('GlobalTestAccuracyRecordforSimplex.mat', 'GlobalAccuracyRecord');
save('GlobalClassTestAccuracyRecordforSimplex.mat', 'GlobalRecording');