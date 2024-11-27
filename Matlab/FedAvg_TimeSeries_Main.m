clc;
clear;
delete(gcp("nocreate"));
%% Define Dataset Path
DatasetPath = fullfile('Dataset_IID2'); 
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
%% 训练结束后，提取特征并使用 t-SNE 可视化

% 定义要提取特征的层
featureLayer = 'fc2';

% 初始化存储所有客户端特征和标签的数组
allFeatures = [];
allLabels = [];
allClients = [];

% 更新类名为 '0', '1', '2', '3'
classes = {'0', '1', '2', '3'};

% 遍历每个参与者（客户端）
for clientNum = 1:6  % 客户端数量为6
    % 获取每个客户端的本地数据集路径
    clientDatasetPath = fullfile('Dataset_IID2', ['local_', num2str(clientNum)]);
    
    % 为客户端创建图像数据存储
    locSet = imageDatastore(clientDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % 将数据拆分为训练集，丢弃其余部分
    [locTrain, ~] = splitEachLabel(locSet, 0.8, "randomized");
    % 进一步拆分，只保留训练数据
    [locTrain, ~] = splitEachLabel(locTrain, 0.875, "randomized");
    
    % 创建增强的图像数据存储
    locTrain = augmentedImageDatastore(inputSize(1:2), locTrain);
    
    % 定义预处理函数
    preprocess = @(X,Y)MiniBatchPreprocessing(X,Y,classes); 
    
    % 为训练数据创建小批量队列
    locTrainMBQ = minibatchqueue(locTrain, ...
        MiniBatchSize = MiniBatchSize, ...
        MiniBatchFcn = preprocess, ...
        MiniBatchFormat = ["SSCB",""]);
    
    % 初始化存储该客户端特征和标签的数组
    clientFeatures = [];
    clientLabels = [];
    
    % 重置小批量队列
    reset(locTrainMBQ);
    
    % 遍历小批量数据
    while hasdata(locTrainMBQ)
        [X, Y] = next(locTrainMBQ);
        
        % 将数据通过网络，获取指定层的特征
        features = forward(globalModel, X, 'Outputs', featureLayer);
        
        % 提取特征
        features = features{1};
        
        % 将特征转换为数值数组并转置
        features = gather(extractdata(features))';
        
        % 将 one-hot 编码的标签解码为分类
        labels = onehotdecode(Y, classes, 1);
        
        % 收集该客户端的特征和标签
        clientFeatures = [clientFeatures; features];
        clientLabels = [clientLabels; labels];
    end
    
    % 将客户端的特征和标签添加到总体数组中
    allFeatures = [allFeatures; clientFeatures];
    allLabels = [allLabels; clientLabels];
    allClients = [allClients; clientNum * ones(size(clientLabels))];
end

% 使用 t-SNE 进行降维
rng('default') % 保持结果可重复
Y = tsne(allFeatures);

% 绘制 t-SNE 结果，按客户端进行颜色区分
figure;
gscatter(Y(:,1), Y(:,2), allClients);
title('客户端数据表示的 t-SNE 可视化');
xlabel('t-SNE 维度 1');
ylabel('t-SNE 维度 2');
legend('客户端 1', '客户端 2', '客户端 3', '客户端 4', '客户端 5', '客户端 6');

% 或者，按标签进行颜色区分
figure;
gscatter(Y(:,1), Y(:,2), allLabels);
title('数据标签的 t-SNE 可视化');
xlabel('t-SNE 维度 1');
ylabel('t-SNE 维度 2');
legend(classes);

