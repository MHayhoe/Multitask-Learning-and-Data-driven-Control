%% Set all constants and parameters
initialize_from_python
mode = 'Min_H';
%alpha_mob = 2*ones(size(alpha_mob));
%parameters.alpha_mob = alpha_mob;
Mult_u = Mult_u(1:T-3, 1:end);

daily_budget = 0.0034226*1.5;

%% The Geometric Program
tic;
Solve_GP_Deaths
toc

%% Plot results
%do_plots_mobility