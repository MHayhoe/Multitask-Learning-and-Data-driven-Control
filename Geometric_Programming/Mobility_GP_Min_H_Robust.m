%% Set all constants and parameters
initialize_from_python
mode = 'Min_H';
Mult_u = Mult_u(1:T-3, 1:end);

daily_budget = 0.11743*2;
alpha_mob=2;

%% The Geometric Program
cvx_solver mosek
cvx_begin gp

variables u(num_vars, num_categories) eta_p(1,T) eta_n(1,T)

% Build beta as a function of u
b = mob_posy(u, c_mob, alpha_mob, b_mob, N, beta_max);

% Build values of state
% Beware indices here: M(t) = M_{t-1}, but X(t) = X_{t}.
M(:,:,1) = [1 - rEI - rEA, S0*b(1), gA*S0*b(1), 0;
            rEI, 1 - rIR, 0, 0;
            rEA, 0, 1 - rAR, 0;
            0, rIH, 0, 1 - rHR - rHD];
X(:,1) = M(:,:,1)*[E0 I0 A0 H0]';
D(1) = D0 + rHD*H0 + b(1)*0;

% Evolve states
for t=2:T
    curr_ind = min(ceil(t/fixed_days), num_vars);
    M(:,:,t) = [1 - rEI - rEA, S0*b(curr_ind), gA*S0*b(curr_ind), 0;
        rEI, 1 - rIR, 0, 0;
        rEA, 0, 1 - rAR, 0;
        0, rIH, 0, 1 - rHR - rHD];
    X(:,t) = M(:,:,t)*X(:,t-1);
    D(t) = D(t-1) + rHD*X(4,t-1);
end
% To easily access H compartment later
H = X(4,:);

% For the discounted objective function
H_discount = discount_factor.^(0:T-1).*(H);

% Compute gradients
H_plus = cell(T,1);
H_minus = cell(T,1);
for t = 3:T
    grad_comp = [];
    for s = 1:t-2
        curr_ind = min(ceil(s/fixed_days), num_vars);
        M_temp = [E0 I0 A0 H0]';
        for l=1:s-1
            M_temp = M(:,:,l)*M_temp;
        end
        M_temp = [0, 1, gA, 0; zeros(3,4)]*M_temp;
        for l=s+1:t
            M_temp = M(:,:,l)*M_temp;
        end
        M_temp = [0,0,0,1]*M_temp;
        grad_comp = [grad_comp; S0*alpha_mob.*c_mob.*u(curr_ind,:).^(alpha_mob - 1) * M_temp];
    end
    plus_inds = repmat(alpha_mob >= 0, t-2, 1);
    H_plus{t} = grad_comp.*plus_inds;
    H_minus{t} = grad_comp.*(~plus_inds);
end


% Create the GP
% Minimize number of deaths given a fixed total budget on cost
minimize(sum(H_discount) + H(end) * infinite_horizon_discount)

subject to

sum(cost_per_category'*(Mult_u*u)'.^(-1)) <= (T-3)*(daily_budget*(1/u_min - 1/u_max) + 1/u_max*sum(cost_per_category));
eps*ones(size(u)) <= u <= ones(size(u));
H + eta_p + eta_n <= tH;
% For norms
for t=1:T
   H_plus{t}'*blkdiag(Sigma{1:t-2})*H_plus{t}*eta_p(t)^(-2) <= chi_sq_quant^(-2);
   H_minus{t}'*blkdiag(Sigma{1:t-2})*H_minus{t}*eta_n(t)^(-2) <= chi_sq_quant^(-2);
end

cvx_end

%% Plot results
do_plots_mobility

