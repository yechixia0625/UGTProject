function [loss, gradient] = FedMOONLossGrad(localModel, globalModel, X, Y, preLocalModel, Temperature, Mu)
    % Supervised loss
    YPred = forward(localModel, X);
    SupLoss = crossentropy(YPred, Y);
    
    % Extracting features
    CurLocalRepresent = predict(localModel, X, 'Outputs', 'fc_embed');
    CurGloRepresent = predict(globalModel, X, 'Outputs', 'fc_embed');
    PreLocRepresent = predict(preLocalModel, X, 'Outputs', 'fc_embed');
    
    % Calculate the cosine similarity
    norm_local = vecnorm(CurLocalRepresent, 2, 1);
    norm_glo = vecnorm(CurGloRepresent, 2, 1);
    norm_pre = vecnorm(PreLocRepresent, 2, 1);
    
    sim1 = sum(CurLocalRepresent .* CurGloRepresent, 1) ./ (norm_local .* norm_glo + 1e-8);
    sim2 = sum(CurLocalRepresent .* PreLocRepresent, 1) ./ (norm_local .* norm_pre + 1e-8);
    
    % Contrastive loss
    ConLoss = -mean(log(exp(sim1/Temperature) ./ (exp(sim1/Temperature) + exp(sim2/Temperature))));
    
    % Total loss
    loss = Mu * ConLoss + SupLoss;
    gradient = dlgradient(loss, localModel.Learnables);
end