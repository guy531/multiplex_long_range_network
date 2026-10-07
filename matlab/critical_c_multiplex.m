%% multiplex_bounds.m
% Calculates:
%   1. Exact rigorous bound C*(sigma) from Theorem 1
%   2. Closed-form bound from Theorem 2
%
% No Symbolic Math Toolbox is required.

clear;
clc;
close all;

%% Sigma values
sigma = linspace(0.02,0.99,150);

Cstar_exact  = zeros(size(sigma));
Cstar_closed = zeros(size(sigma));

%% Number of terms in the logarithmic product expansion
M = 250;

for i = 1:length(sigma)

    s = 1 + sigma(i);

    % z = zeta(1+sigma)
    z = zetaNumeric(s);

    % Precalculate zeta(m*s), because these values are repeatedly used
    % when fzero evaluates different values of C.
    zeta_ms = zeros(1,M);

    for m = 1:M
        zeta_ms(m) = zetaNumeric(m*s);
    end

    %% ---------------------------------------------------------------
    %  1. Exact bound from Theorem 1
    %
    %       2*C*z*kappa(C,sigma) = 1
    % ---------------------------------------------------------------

    fExact = @(C) exactEquation(C,z,zeta_ms);

    bracket = findBracket(fExact);

    Cstar_exact(i) = fzero(fExact, bracket);


    %% ---------------------------------------------------------------
    %  2. Closed-form bound from Theorem 2
    %
    %  2*C*z*[1-(1-C)^z *
    %             (1-C/(1-exp(-2*C*z)))^z] = 1
    % ---------------------------------------------------------------

    fClosed = @(C) closedEquation(C,z);

    bracket = findBracket(fClosed);

    Cstar_closed(i) = fzero(fClosed, bracket);

    fprintf(['sigma = %.4f   C_exact = %.8f   ' ...
             'C_closed = %.8f\n'], ...
             sigma(i), ...
             Cstar_exact(i), ...
             Cstar_closed(i));

end


%% Plot

figure;

plot(sigma,Cstar_exact,'-.' ,'LineWidth',2);
hold on;

plot(sigma,Cstar_closed,'--','LineWidth',2);

xlabel('$\sigma$', 'Interpreter', 'latex');
ylabel('$C_c(\sigma)$', 'Interpreter', 'latex');

load CA.mat; load sigmaA.mat;

plot(sigma, 1./(2*zeta(1+sigma)), '--', 'LineWidth',2);

plot(sigmaA, CA, '.k', 'MarkerSize', 20)

legend('Theorem 1: exact product', ...
       'Theorem 2: closed form', ...
       'Schulman',...
       'Simulation',...
       'Location','northwest');

set(gca, 'FontSize', 20);

grid on;
box on;


%% Optional: save numerical results

results = table(sigma(:), ...
                Cstar_exact(:), ...
                Cstar_closed(:), ...
                'VariableNames', ...
                {'sigma','Cstar_exact','Cstar_closed'});

writetable(results,'multiplex_bounds.csv');


%% ==================================================================
%                       LOCAL FUNCTIONS
% ==================================================================


function F = exactEquation(C,z,zeta_ms)
% Equation:
%
%       F(C) = mu*kappa - 1
%
% where
%
%       mu = 2*C*z
%
%       kappa = 1 - Pplus*Q
%
%       Pplus = product_r (1 - C/r^s)
%
%       Q = product_r (1 - (C/q)/r^s)
%
%       q = 1 - Pplus^2


    %% Calculate log(Pplus)

    logPplus = logInfiniteProduct(C,zeta_ms);

    Pplus = exp(logPplus);


    %% q = 1 - Pplus^2
    %
    % expm1 is more accurate when logPplus is close to zero.

    q = -expm1(2*logPplus);


    %% Calculate second infinite product

    x = C/q;

    if x <= 0 || x >= 1
        F = NaN;
        return
    end

    logQ = logInfiniteProduct(x,zeta_ms);


    %% kappa = 1 - Pplus*Q
    %
    % Pplus*Q = exp(logPplus + logQ)

    kappa = -expm1(logPplus + logQ);


    %% Mean degree

    mu = 2*C*z;


    %% Root condition

    F = mu*kappa - 1;

end



function F = closedEquation(C,z)
% Closed-form equation from Theorem 2:
%
%  2*C*z * [
%       1 - (1-C)^z *
%           (1-C/(1-exp(-2*C*z)))^z
%           ] - 1 = 0


    mu = 2*C*z;

    qLower = -expm1(-mu);
    % qLower = 1 - exp(-2*C*z)


    secondFactor = 1 - C/qLower;

    if secondFactor <= 0
        F = NaN;
        return
    end


    kappaUpper = ...
        1 ...
        - (1-C)^z ...
        * secondFactor^z;


    F = mu*kappaUpper - 1;

end



function logP = logInfiniteProduct(x,zeta_ms)
% Computes
%
%       log[ product_r (1 - x/r^s) ]
%
% using
%
% log P =
%
%   - sum_{m=1}^\infty x^m/m * zeta(m*s).
%
% This avoids truncating the product directly in r.


    M = length(zeta_ms);

    m = 1:M;

    terms = (x.^m)./m .* zeta_ms;

    logP = -sum(terms);

end



function bracket = findBracket(fun)
% Finds the first interval [C1,C2] in which
%
%       fun(C1) < 0
%       fun(C2) > 0.
%
% This is the crossing corresponding to C*.


    Cgrid = linspace(1e-7,0.95,600);

    previousC = NaN;
    previousF = NaN;

    for k = 1:length(Cgrid)

        C = Cgrid(k);
        F = fun(C);

        if ~isfinite(F)
            continue
        end

        if isfinite(previousF)

            if previousF <= 0 && F >= 0

                bracket = [previousC C];
                return

            end

        end

        previousC = C;
        previousF = F;

    end

    error('Could not find a root bracket.');

end



function z = zetaNumeric(s)
% Numerical Riemann zeta function for s > 1.
%
% Uses Euler-Maclaurin summation.
% This is especially important when s is close to 1.

    if s <= 1
        error('zetaNumeric requires s > 1.');
    end

    % For large s, zeta(s) is indistinguishable from 1
    % to double precision apart from the first few terms.
    if s > 50
        z = 1 + 2^(-s) + 3^(-s);
        return
    end

    N = 50;

    n = 1:N-1;

    z = sum(n.^(-s));

    % Euler-Maclaurin tail

    z = z ...
        + N^(1-s)/(s-1) ...
        + 0.5*N^(-s);


    % B_(2k)/(2k)! coefficients
    coeff = [ ...
         1/12, ...
        -1/720, ...
         1/30240, ...
        -1/1209600, ...
         1/47900160, ...
        -691/1307674368000 ...
        ];


    for k = 1:length(coeff)

        order = 2*k - 1;

        rising = 1;

        for j = 0:order-1
            rising = rising*(s+j);
        end

        z = z ...
            + coeff(k) ...
            * rising ...
            * N^(-s-order);

    end

end