function [CurLocalRepresent,gradient] = MOONLossGrad(localModel, globalModel, X, Y, PreLocRepresent, Temperature, Mu)

% supervised loss
YPred = forward(localModel, X);
SupLoss = crossentropy(YPred, Y);

% contrastive loss
CurLocalRepresent = predict(localModel, X, Outputs={'fc3'}); 
CurLocalRepresent = gather(CurLocalRepresent);
CurLocalRepresent = extractdata(CurLocalRepresent);

CurGloRepresent = predict(globalModel, X, Outputs={'fc3'});
CurGloRepresent = gather(CurGloRepresent);
CurGloRepresent = extractdata(CurGloRepresent);

sim1 = dot(CurLocalRepresent, CurGloRepresent) / (norm(CurLocalRepresent) * norm(CurGloRepresent)); 
sim2 = dot(CurLocalRepresent, PreLocRepresent) / (norm(CurLocalRepresent) * norm(PreLocRepresent)); 

ConLoss = -log(exp(sim1/Temperature)/(exp(sim1/Temperature)+exp(sim2/Temperature)));

loss = Mu*ConLoss + SupLoss;
gradient = dlgradient(loss, localModel.Learnables);

end