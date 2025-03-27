% Reconstruct the fc3 weight of the dropout client based on the previous round of information, and update the global fc3 weight using the current alpha aggregation
function [new_global_W, new_global_b, updated_clientW_cell, updated_clientb_cell] = compensateDropoutWeights(...
    alpha, clientW_cell, clientb_cell, drop_client_ids, ...
    prev_global_W, prev_global_b, prev_alpha, prev_clientW_fc3, prev_clientb_fc3, ...
    alphaHistory, clientW_history, clientb_history, currentRound)

    numClients = numel(clientW_cell);
    updated_clientW_cell = clientW_cell;
    updated_clientb_cell = clientb_cell;
    
    for i = 1:numClients
        if ismember(i, drop_client_ids)
            effective_alpha = prev_alpha(i);
            effective_clientW = prev_clientW_fc3{i};
            effective_clientb = prev_clientb_fc3{i};
            effective_round = currentRound - 1;
            % If the α of this client was 0 in the last round, the most recent non-zero round is looked up from the history
            if effective_alpha == 0
                found = false;
                for r = currentRound-1:-1:1
                    if alphaHistory(r, i) > 0
                        effective_alpha = alphaHistory(r, i);
                        effective_clientW = clientW_history{r}{i};
                        effective_clientb = clientb_history{r}{i};
                        effective_round = r;
                        found = true;
                        break;
                    end
                end
                if ~found
                    error('No history of client %d non-zero alpha was found', i);
                end
            end
            fprintf('Client %d compensation uses alpha and weights from round %d, alpha = %f\n', i, effective_round, effective_alpha);
            sumActiveW = zeros(size(prev_global_W));
            sumActiveb = zeros(size(prev_global_b));
            % For the dropout client, refactor using the parameters from the previous round:
            % w_3 =  (1 / alpha_3) * (w_global - alpha_1*w_1 - alpha_2*w_2 - alpha_4*w_4 - alpha_5*w_5 - alpha_6*w_6)
            % the same as
            % w_i = (1/prev_alpha(i))*(prev_global_W - sum_{j∉dropout} prev_alpha(j)*prev_clientW_fc3{j})

            for j = 1:numClients
                if ~ismember(j, drop_client_ids)
                    sumActiveW = sumActiveW + prev_alpha(j) * prev_clientW_fc3{j};
                    sumActiveb = sumActiveb + prev_alpha(j) * prev_clientb_fc3{j};
                end
            end
            
            updated_clientW_cell{i} = (1 / effective_alpha) * (prev_global_W - sumActiveW);
            updated_clientb_cell{i} = (1 / effective_alpha) * (prev_global_b - sumActiveb);
        end
    end
    
    new_global_W = zeros(size(prev_global_W));
    new_global_b = zeros(size(prev_global_b));
    for i = 1:numClients
        new_global_W = new_global_W + alpha(i) * updated_clientW_cell{i};
        new_global_b = new_global_b + alpha(i) * updated_clientb_cell{i};
    end
end