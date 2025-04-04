function newAlpha = computeAlphaFromGradients(gradAll, numClients, Round, CommunicationRounds, simplex_start_epoch)
    persistent pcaCoeff;
    global allSimplexPoints;
    if isempty(allSimplexPoints)
        allSimplexPoints = cell(1, CommunicationRounds);
    end

    if isempty(pcaCoeff)
        % Collect all client gradient data
        allGradients = [];
        for i = 1:numClients
            grad_i = extractdata(gradAll{i});
            allGradients = [allGradients, grad_i];
        end
        [pcaCoeff, ~, ~] = pca(allGradients');
    end
    
    simplexDim = numClients - 1;
    reducedGradients = zeros(simplexDim, numClients);
    for i = 1:numClients
        grad_i = extractdata(gradAll{i});
        % The projection is performed using the first calculated PCA coefficients
        reducedGradients(:,i) = pcaCoeff(:,1:simplexDim)' * grad_i;
    end
    
    s = 0.5;
    lambda = 0.01;
    regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda);
    
    % Project each client gradient onto the simplex
    simplexPoints = zeros(numClients, simplexDim);
    for k = 1:numClients
        simplexPoints(k,:) = project_to_simplex(regularizedGradients(:,k));
    end

    if Round > simplex_start_epoch
        previousSimplexPoints = allSimplexPoints{Round - 1};
        blendingFactor = 0.7;
        for i = 1:numClients
            simplexPoints(i,:) = blendingFactor * previousSimplexPoints(i,:) + (1 - blendingFactor) * simplexPoints(i,:);
        end
    end

    allSimplexPoints{Round} = simplexPoints;

    % Calculate the similarity matrix and sampling weights
    similarityMatrix = pdist(simplexPoints);
    similarityMatrix = squareform(similarityMatrix);
    samplingWeights = 1 ./ (1 + similarityMatrix);
    samplingWeights = samplingWeights ./ sum(samplingWeights, 2);
    
    % Generate a new alpha based on the sample weight
    numSamplesPerClient = 10;
    alpha_weights = zeros(1, numClients);
    for i = 1:numClients
        for rep = 1:numSamplesPerClient
            pick_i = randsample(1:numClients, 1, true, samplingWeights(i,:));
            alpha_weights(pick_i) = alpha_weights(pick_i) + 1;
        end
    end
    newAlpha = alpha_weights / sum(alpha_weights);    
end