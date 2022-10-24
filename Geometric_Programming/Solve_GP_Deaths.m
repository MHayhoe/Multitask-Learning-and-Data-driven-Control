% Solves the minimum deaths GP
cvx_solver mosek
cvx_begin gp

variables u(num_vars, num_categories)   % careful with indices

% Build beta as a function of u
b = mob_posy(u, c_mob, alpha_mob, b_mob, N, beta_max);

% Build values of H
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
Mnew = [1 - rEI - rEA, S0*b(1), gA*S0*b(1);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
X(:,1) = Mnew*[E0 I0 A0]';
H(1) = (1 - rHR - rHD)*H0 + rIH*I0 + b(1)*0;
D(1) = D0 + rHD*H0 + b(1)*0;
H_discount(1) = discount_factor*H(1);
for t=2:T-1
    curr_ind = min(ceil(t/fixed_days), num_vars);
    Mnew = [1 - rEI - rEA, S0*b(curr_ind), gA*S0*b(curr_ind);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
    X(:,t) = Mnew*X(:,t-1);
    H(t) = (1 - rHR - rHD)*H(t-1) + rIH*X(2,t-1);
    D(t) = D(t-1) + rHD*H(t-1);
    H_discount(t) = discount_factor^t*H(t);
end

% Create the GP
% Minimize number of deaths given a fixed total budget on cost
minimize(sum(H_discount) + H(end) * infinite_horizon_discount)

subject to

sum(cost_per_category'*(Mult_u*u)'.^(-1)) <= (T-3)*(daily_budget*(1/u_min - 1/u_max) + sum(cost_per_category));
H <= tH;
u_min*ones(size(u)) <= u <= u_max*ones(size(u));
%u(2:end,:) <= upper_ratio*u(1:end-1,:); % Upper bound as ratio for u
%u(2:end,:) >= lower_ratio*u(1:end-1,:); % Lower bound as ratio for u

cvx_end