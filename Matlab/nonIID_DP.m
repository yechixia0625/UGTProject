clc;
clearvars -except delta noise_scale;
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
%% Define noise_scale
fprintf('noise_scale: %d\n', noise_scale);
fprintf('delta: %d\n', delta);
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
gradient_cutoff = 1.0;

% server published a global model to all participants
localModel = globalModel;

%% Define Privacy Parameters
% 计算各参与者的平均训练样本数
avgTrainSize = mean([locTrainSize{:}]);
q = MiniBatchSize / avgTrainSize; % 采样比例
steps_per_round = LocalEpochs * ceil(avgTrainSize / MiniBatchSize);

% 初始化隐私预算记录
epsilon_values = zeros(CommunicationRounds, 1);

%% Define Monitor
Monitor = trainingProgressMonitor(...
    Metrics = "GlobalAccuracy", ...
    Info="CommunicationRound", ...
    XLabel="Communication Round");
%% Training Circuit
Round = 0;
spmd
    PreLocRepresent = []; % 初始化PreLocRepresent
end

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
                [CurLocalRepresent, gradient] = dlfeval(@FedMOONLossGrad, localModel, globalModel, X, Y, PreLocRepresent, Temperature, Mu);
                numRows = height(gradient); % 先计算行数
                for idx = 1:numRows
                    grad_temp = gradient.Value{idx};
                    % 进行梯度裁剪
                    grad_temp(grad_temp > gradient_cutoff) = gradient_cutoff;
                    grad_temp(grad_temp < -gradient_cutoff) = -gradient_cutoff;
                    % 添加高斯噪声
                    grad_temp = grad_temp + noise_scale * randn(size(grad_temp), 'like', grad_temp);
                    % 更新表中对应的梯度数据
                    gradient.Value{idx} = grad_temp;
                end
                [localModel, Velocity] = sgdmupdate(localModel, gradient, Velocity, LearningRate, Momentum);
                PreLocRepresent = CurLocalRepresent;
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

    % 计算从开始到当前通信轮次的累计梯度更新步数
    cumulative_steps = steps_per_round * Round;
    % 利用 RDP 账户器计算 (ε, δ)-DP 下的 ε
    epsilon_current = computeDPBudget(q, noise_scale, cumulative_steps, delta);
    epsilon_values(Round) = epsilon_current;
    fprintf('Round %d, cumulative_steps=%d, epsilon=%.4f\n', Round, cumulative_steps, epsilon_current);
    % confusion matrix
    plotconfusion(GlobalTestLabel, GlobalTestPred)
    % Create the directory if it doesn't exist
    % outputDir = fullfile(pwd, 'confusionMATRIX');
    % if ~exist(outputDir, 'dir')
    %     mkdir(outputDir);
    % end
    % Save the confusion matrix as an image
    % confusionFileName = fullfile(outputDir, ['ConfusionMatrix_Round_', num2str(Round), '.png']);
    % saveas(gcf, confusionFileName);
    % close(gcf); % Close the figure after saving
    % update monitor
    recordMetrics(Monitor, Round, GlobalAccuracy = GlobalTestAccuracy);
    updateInfo(Monitor, CommunicationRound = Round + " of " + CommunicationRounds);     
    Monitor.Progress = 100 * Round / CommunicationRounds;
end

FinalRoundEachClassAccuracy = GlobalRecording(Round, :);

matFile1 = sprintf('result/GlobalTestAccuracyRecordfornonIID_DP_noise_scale_%d_delta_%g.mat', noise_scale, delta);
matFile2 = sprintf('result/GlobalClassTestAccuracyRecordfornonIID_DP_noise_scale_%d_delta_%g.mat', noise_scale, delta);
matFile3 = sprintf('result/PrivacyBudgetValues_noise_scale_%d_delta_%g.mat', noise_scale, delta);

save(matFile1, 'GlobalAccuracyRecord');
save(matFile2, 'GlobalRecording');
save(matFile3, 'epsilon_values');

figure;
plot(GlobalAccuracyRecord, '-o','LineWidth', 2);
xlabel('Communication Rounds');
ylabel('Global Test Accuracy');
title('Global Test Accuracy over Communication Rounds');
grid on;
pngFile1 = sprintf('result/GlobalTestAccuracy_DP_noise_scale_%d_delta_%g.png', noise_scale, delta);
saveas(gcf, pngFile1);

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
pngFile2 = sprintf('result/GlobalClassTestAccuracy_DP_noise_scale_%d_delta_%g.png', noise_scale, delta);
saveas(gcf, pngFile2);

figure;
plot(1:CommunicationRounds, epsilon_values, '-o','LineWidth', 2);
xlabel('Communication Rounds');
ylabel('Privacy Budget \epsilon');
title('Privacy Budget \epsilon over Communication Rounds');
grid on;
pngFile3 = sprintf('result/PrivacyBudget_vs_Rounds_DP_noise_scale_%d_delta_%g.png', noise_scale, delta);
saveas(gcf, pngFile3);


figure;
plot(epsilon_values, GlobalAccuracyRecord, '-o', 'LineWidth', 2);
xlabel('Privacy Budget \epsilon');
ylabel('Global Test Accuracy');
title('Global Test Accuracy vs Privacy Budget \epsilon');
grid on;
pngFile4 = sprintf('result/GlobalAccuracy_vs_PrivacyBudget_noise_scale_%d_delta_%g.png', noise_scale, delta);
saveas(gcf, pngFile4);