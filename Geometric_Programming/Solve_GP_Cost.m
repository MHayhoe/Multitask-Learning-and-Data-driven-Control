% Solves the minimum cost GP
cvx_solver mosek
cvx_begin gp

variables u(num_vars, num_categories)   % careful with indexes

% Build beta as a function of u
b = mob_posy(u, c_mob(categories), alpha_mob(categories), b_mob, N, beta_max);

% Build values of H
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
Mnew = [1 - rEI - rEA, S0*b(1), gA*S0*b(1);
    rEI, 1 - rIR, 0;
    rEA, 0, 1 - rAR];
X(:,1) = Mnew*[E0 I0 A0]';

expression H(T);
H(1) = (1 - rHR - rHD)*H0 + rIH*I0; % + b(1)*0;

expression D(T);
D(1) = D0 + rHD*H0; % + b(1)*0;
for t=2:T
    curr_ind = ceil(t/fixed_days);
    Mnew = [1 - rEI - rEA, S0*b(curr_ind), gA*S0*b(curr_ind);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
    X(:,t) = Mnew*X(:,t-1);
    H(t) = (1 - rHR - rHD)*H(t-1) + rIH*X(2,t-1);
    D(t) = D(t-1) + rHD*H(t-1);
end

% Create the GP
% Minimize total cost to keep hospital below threshold
minimize(sum(cost_per_category'*(Mult_u(1:T-2,:)*u)'.^(-1)))
 
subject to

H <= tH; % Hospital constraints
u_min*ones(size(u)) <= u <= u_max*ones(size(u)); % Limits on u
u(2:end,:) <= upper_ratio*u(1:end-1,:); % Upper bound as ratio for u
u(2:end,:) >= lower_ratio*u(1:end-1,:); % Lower bound as ratio for u

cvx_end