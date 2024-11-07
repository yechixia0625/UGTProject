%% Loss and Gradient Calculation
function [loss, gradient] = FedAvgLossAndGrad(localModel, X, Y)

    YPred = forward(localModel, X);
    loss = crossentropy(YPred, Y);
    gradient = dlgradient(loss, localModel.Learnables);

end