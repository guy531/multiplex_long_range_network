function [CB, ustar, Pjump, kbar] = CB_mux(sigma)
%CB_MUX  Tree (Bethe) estimate of the mutual-percolation threshold of two
%        coupled one-dimensional long-range layers, p_r = C / r^(1+sigma).
%
%   [CB, ustar, Pjump, kbar] = CB_mux(sigma)
%
%   The self-consistent equation is     P = [ 1 - Pplus(C*P)^2 ]^2 ,
%   with   Pplus(u) = prod_{r>=1} ( 1 - u / r^(1+sigma) ).
%   Writing u = C*P, a non-zero solution exists only for
%
%          C >= CB = min_u  u / ( 1 - Pplus(u)^2 )^2 .
%
%   Outputs
%     CB     threshold value of C
%     ustar  the minimising u  (= CB * Pjump)
%     Pjump  fraction of functional nodes at the threshold
%     kbar   mean degree of one layer at the threshold, 2*CB*zeta(1+sigma)
%
%   Example:   for s = 0.1:0.1:0.9, fprintf('%.1f  %.4f\n', s, CB_mux(s)); end
%
%   No toolbox is required.

    s = 1 + sigma;
    M = 400;                                   % terms of the logarithmic series
    m = 1:M;
    zs = zeros(1, M);
    for k = 1:M
        zs(k) = zetaNumeric(k * s);
    end

    % log Pplus(u) = - sum_m u^m/m * zeta(m*s)      (valid for 0 <= u < 1)
    logPplus = @(u) -sum((u .^ m) ./ m .* zs);
    % Q(u) = 1 - Pplus(u)^2 : probability of at least one successful link
    Q = @(u) -expm1(2 * logPplus(u));
    Cofu = @(u) u / Q(u)^2;

    opts = optimset('TolX', 1e-12);
    [ustar, CB] = fminbnd(Cofu, 1e-6, 0.95, opts);
    Pjump = Q(ustar)^2;
    kbar = 2 * CB * zs(1);
end


function z = zetaNumeric(s)
% Riemann zeta function for s > 1 by Euler-Maclaurin summation.
    if s > 50
        z = 1 + 2^(-s) + 3^(-s);
        return
    end
    N = 50;
    n = 1:N-1;
    z = sum(n .^ (-s)) + N^(1 - s) / (s - 1) + 0.5 * N^(-s);
    coeff = [1/12, -1/720, 1/30240, -1/1209600, 1/47900160, -691/1307674368000];
    for k = 1:length(coeff)
        order = 2 * k - 1;
        rising = 1;
        for j = 0:order-1
            rising = rising * (s + j);
        end
        z = z + coeff(k) * rising * N^(-s - order);
    end
end
