%% Federated Averaging
function gloLearnable = FederatedAveraging(locFactor, locLearnable)

    % initialise template
    template = locLearnable{1};  
    gloLearnable = cell(height(template), 1); 

    for i = 1:height(gloLearnable)   
        gloLearnable{i} = zeros(size(template{i}), "like", template{i});
    end

    % accumulate local model parameters
    for j = 1:size(locFactor, 2)
        template = locLearnable{j}; 
        for k = 1:numel(gloLearnable)
            gloLearnable{k} = gloLearnable{k} + locFactor(j) .* template{k};
        end
    end
    
end