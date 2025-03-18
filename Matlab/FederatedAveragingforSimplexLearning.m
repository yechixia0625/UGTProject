function gloLearnable = FederatedAveragingforSimplexLearning(locFactor, locLearnable)

    template = locLearnable{1};  
    numLearnables = height(template);
    gloValues = cell(numLearnables, 1);
    
    for i = 1:numLearnables
        currentValue = template.Value{i};
        gloValues{i} = zeros(size(currentValue), "like", currentValue);
    end
    
    numClients = numel(locFactor);
    for j = 1:numClients
        currentTable = locLearnable{j}; 
        for k = 1:numLearnables
            currentValue = currentTable.Value{k};
            gloValues{k} = gloValues{k} + locFactor(j) .* currentValue;
        end
    end
    
    template.Value = gloValues;
    gloLearnable = template;
end
