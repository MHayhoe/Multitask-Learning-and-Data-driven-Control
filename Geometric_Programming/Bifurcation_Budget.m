% Solves the minimal deaths GP using a number of different budgets
%% Initialize
% Load all data
initialize_from_python

% Set budgets to use
Budgets = 1:1:10;
num_budgets = length(Budgets);

% Set hospitalization thresholds to use
Thresholds = 5279*(0.3:0.1:1);
num_thresholds = length(Thresholds);

% Initialize variables
H_cell = cell(num_budgets, num_thresholds);
D_cell = cell(num_budgets, num_thresholds);
u_cell = cell(num_budgets, num_thresholds);
H_final = zeros(num_budgets, num_thresholds);
H_pred_final = zeros(num_budgets, num_thresholds);
D_final = zeros(num_budgets, num_thresholds);
D_pred_final = zeros(num_budgets, num_thresholds);
solve_time = zeros(num_budgets, num_thresholds);

%% Loop through the budgets and hospitalization thresholds
for tH_ind = 1:num_thresholds
    % Find minimal cost
    Mult_u = zeros(T,num_vars);
    for tt = 1:T-1
       Mult_u(tt,ceil(tt/fixed_days)) = 1; 
    end
    mode = 'Min_Cost';
    Solve_GP_Cost;
    min_daily_budget = sum(cost_per_category'*((Mult_u(1:T-2,:)*u)'.^(-1) - 1/u_max))/((T-1)*(1/u_min - 1/u_max));
    
    % Set up for minimal deaths GP
    Mult_u = Mult_u(1:T-3, 1:end);
    mode = 'Min_H';
    clear X M D H H_discount;
    
    % Loop through budgets
    for B_ind = 1:num_budgets
        tic;
        daily_budget = min_daily_budget*Budgets(B_ind);
        tH = Thresholds(tH_ind);
        Solve_GP_Deaths;
        H_cell{B_ind, tH_ind} = H;
        D_cell{B_ind, tH_ind} = D;
        u_cell{B_ind, tH_ind} = u;
        H_final(B_ind, tH_ind) = H_cell{B_ind, tH_ind}(end);
        D_final(B_ind, tH_ind) = D_cell{B_ind, tH_ind}(end);
        [D_predicted, H_predicted] = make_trajectory(parameters, [Mult_u*u; zeros(3, num_mob_categories)]);
        H_pred_final(B_ind, tH_ind) = H_predicted(end);
        D_pred_final(B_ind, tH_ind) = D_predicted(end);
        solve_time(B_ind, tH_ind) = toc

        % Remove variables that are converted to double from CVX
        clear X M D H H_discount;
    end
end
% figure; hold on;
% plot(Budgets, D_final)
% plot([min_daily_budget, min_daily_budget], ylim, '--k')

%% Plot results
% Deaths
fontSize = 24;
tickFontSize = 20;
figure(1); clf; hold on; grid on;
set(gca, 'FontSize', tickFontSize);
sD = surf(Thresholds, Budgets, D_final - D_pred_final);
title('Difference $D(T_c) - \tilde{D}(T_c)$','Interpreter','latex','fontsize',fontSize)
xlabel('Hosp. threshold $\tau_H$','Interpreter','latex','fontsize',fontSize)
ylabel('Multiple of budget $\mathcal{B}$','interpreter','latex','fontsize',fontSize)
xlim([min(Thresholds) max(Thresholds)])
ylim([min(Budgets) max(Budgets)])
set(gca,'TickLabelInterpreter','latex')
ytl = {'$2\times$', '$4\times$', '$6\times$','$8\times$', '$10\times$'};
yticklabels(ytl);
camorbit(-135,-75);

% Hospitalizations
fontSize = 24;
tickFontSize = 20;
figure(2); clf; hold on; grid on;
set(gca, 'FontSize', tickFontSize);
surf(Thresholds, Budgets, H_final - H_pred_final)
title('Difference $H(T_c) - \tilde{H}(T_c)$','Interpreter','latex','fontsize',fontSize)
xlabel('Hosp. threshold $\tau_H$','Interpreter','latex','fontsize',fontSize)
ylabel('Multiple of budget $\mathcal{B}$','interpreter','latex','fontsize',fontSize)
xlim([min(Thresholds) max(Thresholds)])
ylim([min(Budgets) max(Budgets)])
set(gca,'TickLabelInterpreter','latex')
yticklabels(ytl);
camorbit(-135,-75);