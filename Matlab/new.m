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
    
    dropout_round = 30;
    drop_client_id = 6;
    

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
    
    GlobalRecording = zeros(CommunicationRounds, NumClasses);
    GlobalAccuracyRecord = zeros(1, CommunicationRounds);
    
    while Round < CommunicationRounds && ~Monitor.Stop 
        Round = Round + 1;
        
        spmd
            if (Round >= dropout_round) && (spmdIndex == drop_client_id)
                % DropOut: Skip the local training process without calculating the gradient or updating the model
                localModel.Learnables.Value = globalModel.Learnables.Value;
                locLearnable = localModel.Learnables.Value;
                if Round > 20
                    numLayers = height(globalModel.Learnables);
                    dummyLossGradient.Value = cell(numLayers, 1);
                    for row = 1:numLayers
                        zero_tensor = zeros(size(globalModel.Learnables.Value{row}), 'like', globalModel.Learnables.Value{row});
                        dummyLossGradient.Value{row} = dlarray(zero_tensor);
                    end
                    locLossGradient = dummyLossGradient;
                end
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
                        if Round > 20
                            lastGradient = gradient;
                        end
                    end
                end
                locLearnable = localModel.Learnables.Value;
                if Round > 20
                    locLossGradient = lastGradient;
                end
            end
        end
        
        % Select the aggregation mode based on the current round
        if Round <= 20
            %% The first 20 rounds used Federated Averaging
            % Calculate the proportion of samples of each client, and set the number of samples of offline clients to 0
            locTrainSizeCell = cell(1, participants);
            for k = 1:participants
                locTrainSizeCell{k} = locTrainSize{k};
            end
            if Round >= dropout_round
                locTrainSizeCell{drop_client_id} = 0;
            end
            locFactor = [locTrainSizeCell{:}] / sum([locTrainSizeCell{:}]);
            globalModel.Learnables.Value = FederatedAveraging(locFactor, locLearnable);
        else
            %% Use the Simplex Learning for round 21 and beyond
            numClients = participants;
            lastFCLayerName = 'fc3';
            numLayers = height(globalModel.Learnables);
            allGradients = [];  
    
            % Collects the gradient of the last fully connected layer of each client and expands it as a vector
            for k = 1:numClients
                gradVector = [];
                for row = 1:numLayers
                    if strcmp(globalModel.Learnables.Layer(row), lastFCLayerName)
                        gradRow = locLossGradient{k}.Value{row};
                        gradRow = extractdata(gradRow);
                        gradRow = double(gradRow);
                        gradVector = [gradVector; gradRow(:)];
                    end
                end
                allGradients = [allGradients, gradVector];
            end
    
            % PCA reduces the dimension to numClients-1
            simplexDim = numClients - 1;
            [coeff, ~, ~] = pca(double(allGradients'));
            topPC = coeff(:, 1:simplexDim);
            reducedGradients = topPC' * allGradients; 
    
            % Apply Riesz s-Energy regularization
            s = 0.5;
            lambda_riesz = 0.01;
            regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda_riesz);
    
            % Project onto simplex
            simplexPoints = zeros(numClients, simplexDim);
            for k = 1:numClients
                simplexPoints(k,:) = project_to_simplex(regularizedGradients(:,k));
            end
    
            % Calculate the similarity between clients and sample alpha
            distMatrix = pdist2(simplexPoints, simplexPoints);
            simMatrix = 1 ./ (1 + distMatrix);
            simMatrix = simMatrix ./ sum(simMatrix, 2);
    
            alphaCount = zeros(1, numClients);
            for k = 1:numClients
                chosen = randsample(1:numClients, 1, true, simMatrix(k,:));
                alphaCount(chosen) = alphaCount(chosen) + 1;
            end
            alpha = alphaCount / sum(alphaCount);
    
            % Update global model parameters:
            % The weighted average based on alpha was used for the fc3 layer, and the rest of the layers were updated by normal gradient descent
            for row = 1:numLayers
                if strcmp(globalModel.Learnables.Layer(row), 'fc3')
                    aggregatedGradient = 0;
                    for k = 1:numClients
                        clientGrad = locLossGradient{k}.Value{row};
                        aggregatedGradient = aggregatedGradient + alpha(k) * clientGrad;
                    end
                    globalModel.Learnables.Value{row} = globalModel.Learnables.Value{row} - LearningRate * aggregatedGradient;
                else
                    aggregatedGradient = 0;
                    for k = 1:numClients
                        clientGrad = locLossGradient{k}.Value{row};
                        aggregatedGradient = aggregatedGradient + clientGrad;
                    end
                    aggregatedGradient = aggregatedGradient / numClients;
                    globalModel.Learnables.Value{row} = globalModel.Learnables.Value{row} - LearningRate * aggregatedGradient;
                end
            end
        end
    
        %% Global Model Evaluation 
        [GlobalTestAccuracy, GlobalTestLabel, GlobalTestPred, GlobalClassAccuracy] = ...
            EvaluateModelForSixClients(globalModel, gloTsetMBQ, classes);
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