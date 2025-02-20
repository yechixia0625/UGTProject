%% Loss and Gradient Calculation
function [loss, gradient] = fedavg_loss_gradient_calc(localModel, X, Y)

    YPred = forward(localModel, X);
    loss = crossentropy(YPred, Y);
    gradient = dlgradient(loss, localModel.Learnables);

end