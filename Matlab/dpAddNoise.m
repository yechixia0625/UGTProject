% dpAddNoise.m
function gOut = dpAddNoise(gIn, C, sigma)
    % 对单层梯度执行 L2 裁剪并添加 sigma*C 高斯噪声
    % gIn   —— dlarray   单层梯度
    % C     —— double    裁剪阈值
    % sigma —— double    噪声倍率
    % gOut  —— dlarray   处理后的梯度
    
    % L2 范数
    gradNorm = sqrt(sum(gIn.^2, 'all'));
    
    % 裁剪系数
    clipFactor = min(1, C / (gradNorm + eps));
    
    % 裁剪
    gClipped = gIn * clipFactor;
    
    % 生成同形状高斯噪声
    noise = sigma * C * randn(size(gIn), 'like', gIn);
    
    % 加噪
    gOut = gClipped + noise;
end
