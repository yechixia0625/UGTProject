function simplexPoint = project_to_simplex(v)
    % Project a vector v onto the standard simplex
    % 1) Ensure non-negativity
    % 2) Sum to 1
    v_proj = v;
    % If all elements are < 0, handle this degenerate case
    if all(v < 0)
        v_proj = zeros(size(v));
        v_proj(1) = 1;
        simplexPoint = v_proj;
        return;
    end
    
    while sum(v_proj) <= 0 || any(v_proj < 0)
        v_proj = max(0, v_proj);
        s = sum(v_proj);
        if s > 0
            v_proj = v_proj / s;
        else
            v_proj = zeros(size(v));
            v_proj(1) = 1;
            break;
        end
    end
    
    simplexPoint = v_proj;
end