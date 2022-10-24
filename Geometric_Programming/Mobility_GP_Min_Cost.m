% Set all constants and parameters
initialize_from_python
mode = 'Min_Cost';
%alpha_mob = 2*ones(size(alpha_mob));
%parameters.alpha_mob = alpha_mob;

%% The Geometric Program
tic;
Solve_GP_Cost;
toc

daily_budget = sum(cost_per_category'*((Mult_u(1:T-2,:)*u)'.^(-1) - 1/u_max))/((T-1)*(1/u_min - 1/u_max));
disp(['Average daily cost ' num2str(daily_budget)])

%% Plot results
%do_plots_mobility