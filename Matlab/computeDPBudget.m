function epsilon = computeDPBudget(q, noise_multiplier, steps, delta)
    % computeDPBudget 使用 RDP 账户器计算 (ε, δ)-DP 下的 ε
    %
    % 输入：
    %   q               - 采样比例 (例如 MiniBatchSize 除以本地训练集大小)
    %   noise_multiplier- 噪声倍数，即 sigma
    %   steps           - 迭代次数（mini-batch 更新的次数）
    %   delta           - 目标 δ 值（例如 1e-5）
    %
    % 输出：
    %   epsilon         - 根据 RDP 账户器计算出的隐私预算 ε

    % 选择一组 Rényi 阶数
    orders = [1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30, 40, 64];
    rdp = zeros(size(orders));
    % 对于每个 Rényi 阶数计算总的 RDP
    for i = 1:length(orders)
        alpha = orders(i);
        rdp(i) = compute_rdp(q, noise_multiplier, steps, alpha);
    end
    % 将 RDP 转换为 (ε, δ)-DP 中的 ε，公式为：
    %   ε = min_{α} { RDP(α) - log(δ*(α-1))/(α-1) }
    epsilons = rdp - log(delta*(orders - 1)) ./ (orders - 1);
    epsilon = min(epsilons);
end

function rdp = compute_rdp(q, sigma, steps, alpha)
    % compute_rdp 计算单步采样高斯机制在 Rényi 阶数 alpha 下的 RDP 上界，
    % 并对训练过程中的所有迭代进行线性叠加。
    %
    % 输入：
    %   q     - 采样比例
    %   sigma - 噪声倍数
    %   steps - 迭代次数（mini-batch 更新次数）
    %   alpha - Rényi 阶数（α > 1）
    %
    % 输出：
    %   rdp   - 对应阶数下累计的 RDP 值
    %
    % 采用的上界（参考文献中经典的分析）为：
    %   rdp_single = 1/(α-1)*log(1 + q^2 * (α*(α-1)/2) * min( 4*(exp(1/sigma^2)-1), exp((α-1)/sigma^2) ))
    
    if q == 0
        rdp = 0;
        return;
    end
    if sigma == 0
        rdp = Inf;
        return;
    end
    % 计算两个候选项
    term1 = 4 * (exp(1/sigma^2) - 1);
    term2 = exp((alpha - 1) / sigma^2);
    rdp_single = (1 / (alpha - 1)) * log(1 + q^2 * (alpha * (alpha - 1) / 2) * min(term1, term2));
    % 线性叠加所有迭代的 RDP
    rdp = steps * rdp_single;
end
