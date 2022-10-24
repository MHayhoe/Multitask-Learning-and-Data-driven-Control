function z = mob_posy(x, c, alpha, b, n, beta_max)
    % x is 1-by-T, c is N-by-1, alpha is N-by-1, b is a scalar
    z = (sum(repmat(c, size(x,1), 1).*x.^repmat(alpha, size(x,1), 1), 2) + b) / n * beta_max;