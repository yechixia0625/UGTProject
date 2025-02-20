function alpha = random_simplex_point(n)
    % Generate a random point on the simplex shape
    k = n+1;
    u = rand(1,k);
    v = -log(u);
    v_sort = sort(v);
    lambda = v_sort(k);
    alpha = exp(v_sort - lambda);
    alpha = alpha/sum(alpha);
end