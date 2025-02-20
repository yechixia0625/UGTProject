function regularizedGradients = reisz_energy_regularization(reducedGradients, s, lambda)
% Riesz s-Energy Regularization: Ensures a regular spread of projected
% points across the simplex space


numClients = size(reducedGradients, 2); % Number of clients
gradientDim = size(reducedGradients, 1); % Dimension of reduced gradients

regularizedGradients = zeros(size(reducedGradients));

for i = 1:numClients
    grad_i = reducedGradients(:, i);
    energy_term = zeros(gradientDim, 1);

    for j = 1:numClients
        if i ~= j
            grad_j = reducedGradients(:, j);
            diff = grad_i - grad_j;
            distance = norm(diff);

            if distance > 0
                energy_term = energy_term + (diff / (distance^(s+1)));
            end

        end
    end
    regularizedGradients(:, i) = grad_i - lambda * energy_term;
end

end