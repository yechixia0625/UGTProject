function simplexPoint = project_to_simplex(v)
    v = max(0, v);
    if sum(v) > 0
        simplexPoint = v / sum(v);
    else
        v(1) = 1;
        simplexPoint = v;
    end
end
