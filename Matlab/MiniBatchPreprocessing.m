%% Preprocess Minibatch
function [X, Y] = MiniBatchPreprocessing(X, Y, classes)

    % [height, width, channels, batch size]
    X = cat(4, X{1:end}); 

    % [label, batch size]
    Y = cat(2, Y{1:end}); 
    Y = onehotencode(Y, 1, ClassNames = classes);
    
end