function simplexPoint = project_to_simplex(v)
    % Projecting a vector v onto the standard simplex

    v_proj = v;
    % Check if all elements are negative (to avoid div. 0)
    if all(v < 0)  
        % Setting the first element to 1
        v_proj(1) = 1;
    end
    
    while sum(v_proj) <= 0 || any(v_proj < 0)
        v_proj = max(0, v_proj);
        v_proj = v_proj / sum(v_proj);
    end
    
    simplexPoint = v_proj;

end