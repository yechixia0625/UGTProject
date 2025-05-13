clc;
clear;
close all;
%% Define the path of the dataset
DatasetPath = fullfile('Dataset_IID'); 

%% Merge all client data
clientFolders = dir(fullfile(DatasetPath, 'local_*'));
clientFolders = {clientFolders.name};

allImages = {};
allLabels = {};

classNames = {'0', '1', '2', '3'};

for i = 1:length(clientFolders)
    clientPath = fullfile(DatasetPath, clientFolders{i});
    for j = 1:length(classNames)
        classPath = fullfile(clientPath, classNames{j});
        if isfolder(classPath)
            imageFiles = dir(fullfile(classPath, '*.png'));
            for k = 1:length(imageFiles)
                imgPath = fullfile(classPath, imageFiles(k).name);
                allImages{end+1} = imgPath;
                allLabels{end+1} = classNames{j};
            end
        end
    end
end

%% Create an image data store
fullSet = imageDatastore(allImages, 'Labels', categorical(allLabels));

%% Define the image size
inputSize = [160 20 1]; 

%% Dataset division (70% training, 15% validation, 15% testing)
[trainSet, tempSet] = splitEachLabel(fullSet, 0.7, 'randomized');
[valSet, testSet] = splitEachLabel(tempSet, 0.5, 'randomized');

%% Data enhancement
augmentedTrainSet = augmentedImageDatastore(inputSize(1:2), trainSet);
augmentedValSet = augmentedImageDatastore(inputSize(1:2), valSet);
augmentedTestSet = augmentedImageDatastore(inputSize(1:2), testSet);

%% Obtain dataset information
classes = categories(trainSet.Labels);
numClasses = numel(classes);

%% Define the network structure
layers = [
    imageInputLayer(inputSize, Normalization = "none")
    
    convolution2dLayer([5 1], 16, 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool1')
    
    convolution2dLayer([5 1], 16, 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 1], 'Stride', [2 1], 'Name', 'maxpool2')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu4')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')];

net = dlnetwork(layers);

%% Training parameter Settings
epochs = 50;
miniBatchSize = 100;
learningRate = 0.0001;
momentum = 0.5;
velocity = [];

%% Create minibatchqueue
preprocess = @(X,Y)preprocessMiniBatch(X,Y,classes);

trainMBQ = minibatchqueue(augmentedTrainSet,...
    MiniBatchSize = miniBatchSize,...
    MiniBatchFcn = preprocess,...
    MiniBatchFormat = ["SSCB",""]);

valMBQ = minibatchqueue(augmentedValSet,...
    MiniBatchSize = miniBatchSize,...
    MiniBatchFcn = preprocess,...
    MiniBatchFormat = ["SSCB",""]);

testMBQ = minibatchqueue(augmentedTestSet,...
    MiniBatchSize = miniBatchSize,...
    MiniBatchFcn = preprocess,...
    MiniBatchFormat = ["SSCB",""]);

%% Training monitoring Settings
monitor = trainingProgressMonitor(...
    Metrics = "TestAccuracy",...
    Info = "Epoch",...
    XLabel = "Epoch");

%% Traning Loop
velocity = [];
testAccuracies = zeros(1, epochs);
testClassAccuracies = zeros(epochs, numClasses);
for epoch = 1:epochs
    % Training
    [net, velocity, trainLoss, trainAcc] = processEpoch(net, trainMBQ, velocity, learningRate, momentum, classes);
    reset(testMBQ);
    % Validation
    [~, testAcc, testClassAcc] = modelValidation(net, testMBQ, classes);
    testAccuracies(epoch) = testAcc; 
    testClassAccuracies(epoch, :) = testClassAcc;
    fprintf('Epoch %3d / %3d  |  Test Accuracy = %.4f\n', epoch, epochs, testAcc);
    % Record indicators
    recordMetrics(monitor, epoch, TestAccuracy = testAcc);
    updateInfo(monitor, Epoch = epoch + " of " + epochs);
    monitor.Progress = 100 * epoch / epochs;
end
%% Save the test accuracy rate
save(fullfile('test_accuracy.mat'), 'testAccuracies');
save('test_class_accuracy.mat', 'testClassAccuracies', 'classes');

%% Draw the test accuracy
figure;
plot(testAccuracies, 'g', 'LineWidth', 2);
xlabel('Epoch');
ylabel('Test Accuracy');
legend('CNN');
grid on;
saveas(gcf, 'AccuracyCurve.png');

%% Support Function
function [X, Y] = preprocessMiniBatch(XCell, YCell, classes)
    % Preprocess image data
    X = cat(4, XCell{:});
    X = dlarray(single(X), 'SSCB');
    
    % Preprocess label data
    Y = cat(2, YCell{:});
    Y = onehotencode(Y, 1, 'ClassNames', classes);
    Y = dlarray(Y, 'CB');
end

function [loss, gradients] = FedAvgLossGrad(net, X, Y)
    YPred = forward(net, X);
    loss = crossentropy(YPred, Y);
    gradients = dlgradient(loss, net.Learnables);
end

function [net, velocity, epochLoss, epochAcc] = processEpoch(net, mbq, velocity, lr, momentum, classes)
    epochLoss = 0;
    correct = 0;
    total = 0;
    numBatches = 0;
    
    shuffle(mbq);
    
    while hasdata(mbq)
        [X, Y] = next(mbq);
        [loss, grads] = dlfeval(@FedAvgLossGrad, net, X, Y);
        [net, velocity] = sgdmupdate(net, grads, velocity, lr, momentum);
        
        YPred = predict(net, X);
        YPred = extractdata(YPred);
        [~, YPred] = max(YPred, [], 1);
        YPred = double(YPred);
        
        Y_onehot = extractdata(Y);
        YTrue = onehotdecode(Y_onehot, classes, 1);
        YTrue = double(YTrue);
        
        correct = correct + sum(YPred == YTrue);
        total = total + numel(YTrue);
        
        epochLoss = epochLoss + extractdata(loss);
        numBatches = numBatches + 1;
    end
    
    epochLoss = epochLoss / numBatches;
    epochAcc = correct / total;
end

function [loss, acc, classAcc] = modelValidation(net, mbq, classes)
    reset(mbq);
    numClasses = numel(classes);

    loss = 0;
    correct = 0;  total = 0;  numBatches = 0;

    classCorrect = zeros(1, numClasses);
    classTotal   = zeros(1, numClasses);

    while hasdata(mbq)
        [X, Y] = next(mbq);
        YPred = forward(net, X);
        loss  = loss + extractdata(crossentropy(YPred, Y));
        YPred = extractdata(YPred);
        [~, YPred] = max(YPred,[],1);
        YPred = double(YPred);
        Y_true_onehot = extractdata(Y);
        YTrue = double(onehotdecode(Y_true_onehot, classes, 1));
        correct = correct + sum(YPred == YTrue);
        total   = total   + numel(YTrue);
        for c = 1:numClasses
            idx = (YTrue == c);
            classTotal(c)   = classTotal(c)   + sum(idx);
            classCorrect(c) = classCorrect(c) + sum(YPred(idx) == c);
        end
        numBatches = numBatches + 1;
    end
    loss = loss / numBatches;
    acc = correct / total;
    classAcc = classCorrect ./ max(classTotal,1);
end

function [accuracy, trueLabels, predLabels, classAcc] = modelEvaluation(net, mbq, classes)
    trueLabels = [];
    predLabels = [];
    
    while hasdata(mbq)
        [X, Y] = next(mbq);
        YPred = predict(net, X);
        YPred = extractdata(YPred);
        [~, batchPred] = max(YPred, [], 1);
        batchPred = double(batchPred);
        Y_onehot = extractdata(Y);
        batchTrue = onehotdecode(Y_onehot, classes, 1);
        batchTrue = double(batchTrue);
        trueLabels = [trueLabels batchTrue];
        predLabels = [predLabels batchPred];
    end
    
    trueLabels = double(trueLabels);
    predLabels = double(predLabels);
    
    accuracy = sum(predLabels == trueLabels) / numel(trueLabels);
end