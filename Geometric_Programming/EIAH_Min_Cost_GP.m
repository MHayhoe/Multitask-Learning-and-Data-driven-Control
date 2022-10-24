% Set all constants and parameters
initialize_GP

%% The Geometric Program
cvx_solver mosek
cvx_begin gp

variables b(num_vars)   % careful with indexes

% Build values of I
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
Mnew = [1 - rEI - rEA, S0*b(1), gA*S0*b(1);
    rEI, 1 - rIR, 0;
    rEA, 0, 1 - rAR];
X(:,1) = Mnew*[E0 I0 A0]';
H(1) = (1 - rHR - rHD)*H0 + rIH*I0 + b(1)*0;
for t=2:T-1
    curr_ind = ceil(t/fixed_days);
    Mnew = [1 - rEI - rEA, S0*b(curr_ind), gA*S0*b(curr_ind);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
    X(:,t) = Mnew*X(:,t-1);
    H(t) = (1 - rHR - rHD)*H(t-1) + rIH*X(2,t-1);
end

% Create the GP
% Minimize total cost to keep hospital below threshold
minimize(sum((Mult_b(1:T-2,:)*b).^(-1))) % + sum((b(2:end-1)./b(1:end-2) + b(1:end-2)./b(2:end-1)).^(1/2)))

subject to

H <= tH;
%H(end) <= 00;
blow <= b <= bhigh;

cvx_end

disp(['Average daily cost ' num2str((sum((Mult_b(1:T-2,:)*b).^(-1)) - 1/bhigh)/(1/blow - 1/bhigh)/(T-2))])

%% Plot results
do_plots