function gloLearnable = FederatedAveragingforSimplexLearning(locFactor, locLearnable)
    % 从第一个客户端的 Learnables 表中初始化模板
    template = locLearnable{1};  
    numLearnables = height(template);
    gloValues = cell(numLearnables, 1);
    
    % 初始化聚合后的参数值
    for i = 1:numLearnables
        currentValue = template.Value{i}; % 通过 .Value 访问表中的数据
        gloValues{i} = zeros(size(currentValue), "like", currentValue);
    end
    
    % 累计各客户端的模型参数
    numClients = numel(locFactor);
    for j = 1:numClients
        currentTable = locLearnable{j}; 
        for k = 1:numLearnables
            currentValue = currentTable.Value{k};
            gloValues{k} = gloValues{k} + locFactor(j) .* currentValue;
        end
    end
    
    % 更新模板表中的 Value 列，并返回聚合结果
    template.Value = gloValues;
    gloLearnable = template;
end
