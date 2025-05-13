function gOut = dpAddNoise(gIn, C, sigma)

    gradNorm = sqrt(sum(gIn.^2, 'all'));
    
    clipFactor = min(1, C / (gradNorm + eps));
    
    gClipped = gIn * clipFactor;
    
    noise = sigma * C * randn(size(gIn), 'like', gIn);
    
    gOut = gClipped + noise;
end
