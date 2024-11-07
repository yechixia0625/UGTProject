 function [Accuracy, Label, Pred]  = EvaluateModel(globalModel, MBQ, classes)

    Label = []; 
    Pred = [];
    correctPred = [];

    % data shuffle
    shuffle(MBQ);
    % mini-batch learning
    while hasdata(MBQ)
        [X, Y] = next(MBQ);

        % label
        YLabel = onehotdecode(Y, classes, 1)'; 
        Label = [Label; YLabel];

        % predict
        YPred = predict(globalModel, X); 
        YPred = onehotdecode(YPred, classes, 1)'; 
        Pred = [Pred; YPred]; 

        correctPred = [correctPred; YPred == YLabel];

    end

    % overall accuracy
    Accuracy = single(sum(correctPred) / size(correctPred, 1));
    
end