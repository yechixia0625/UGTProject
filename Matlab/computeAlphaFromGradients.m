function newAlpha = computeAlphaFromGradients(gradAll, numClients)
    % Collect all client gradients
    allGradients = [];
    for i = 1:numClients
        grad_i = extractdata(gradAll{i});
        allGradients = [allGradients, grad_i];
    end
    % Performing PCA
    [coeff, ~, ~] = pca(allGradients);
    simplexDim = numClients - 1;
    reducedGradients = coeff(:,1:simplexDim)';
    
    % Regularization
    s = 0.5;
    lambda = 0.01;
    regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda);
    
    % Project onto simplex
    simplexPoints = zeros(numClients, simplexDim);
    for k = 1:numClients
        simplexPoints(k,:) = project_to_simplex(regularizedGradients(:,k));
    end
    
    % Calculate the similarity matrix and sampling weights
    similarityMatrix = pdist(simplexPoints);
    similarityMatrix = squareform(similarityMatrix);
    samplingWeights = 1 ./ (1 + similarityMatrix);
    samplingWeights = samplingWeights ./ sum(samplingWeights, 2);
    
    % Generate a new alpha based on the sample weight
    alpha_weights = zeros(1, numClients);
    for i = 1:numClients
        pick_i = randsample(1:numClients, 1, true, samplingWeights(i,:));
        alpha_weights(pick_i) = alpha_weights(pick_i) + 1;
    end
    newAlpha = alpha_weights / sum(alpha_weights);
end
