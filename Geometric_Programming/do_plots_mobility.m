% Set parameters
fontSize = 24;
tickFontSize = 20;
legendFontSize = 14;
contentType = 'image';
prefix = 'Plots/';

%% For the dates
date_offset = datenum('22-Jan-2020');
start_deaths = 192;
dates = start_deaths+date_offset:start_deaths+date_offset+T;
x_tick_inds = [dates(1),dates(32),dates(62)];%,dates(93)];

%% Cost
figure(1); clf; hold on;
plot(dates(1:size(Mult_u,1)),cost_per_category'*((Mult_u*u)'.^(-1) - 1/u_max)/(1/u_min - 1/u_max),'linewidth',3)
if ~strcmp(mode, 'Min_Cost')
    plot(dates(1:size(Mult_u,1)),daily_budget*ones(size(Mult_u,1),1),'--r','linewidth',3)
end
set(gca, 'FontSize', tickFontSize);
title('Total cost $C_t(\mathbf{u}(t))$','Interpreter','latex','FontSize',fontSize)
xlabel('Date','Interpreter','latex','FontSize',fontSize)
ylabel('Cost','Interpreter','latex','FontSize',fontSize)
if strcmp(mode, 'Min_Cost')
    xlim([dates(1) dates(end-4)])
else
    xlim([dates(1) dates(end-5)])
end
xticks(x_tick_inds)
datetick('x','mmm dd','keepticks','keeplimits')
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, [prefix '' mode '_Cost.fig'])
exportgraphics(gca, [prefix '' mode '_Cost.eps'], 'ContentType', contentType)

%% Value of u
eps_mob = 0.001;
%max_mob = max(max(mob_data_raw));
%min_mob = min(min(mob_data_raw));
%u_scaled = Mult_u*u*(max_mob - min_mob - eps_mob) + min_mob - eps_mob;
u_scaled = Mult_u*u;

figure(4); clf; hold on;
Markers = {'o', '*', 'x', '^', 'diamond'};
for m_ind = 1:size(mobility_data,2)
    plot(dates(1:size(Mult_u,1)),u_scaled(:,m_ind)*100,'linewidth',2,'marker',Markers{m_ind},'MarkerSize',8,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
end
%plot(dates(1:size(Mult_u,1)),mob_data_raw(:,end-T+1:end),'linewidth',3,'linestyle','--')
set(gca, 'FontSize', tickFontSize);
title('Value of mobility control action $\mathbf{u}(t)$','Interpreter','latex','FontSize',fontSize)
% plot(ones(size(u))*eps,'--r')
xlabel('Date','Interpreter','latex','FontSize',fontSize)
ylabel('Mobility control actions $u_k$','Interpreter','latex','FontSize',fontSize)
if strcmp(mode, 'Min_Cost')
    xlim([dates(1) dates(end-4)])
else
    xlim([dates(1) dates(end-5)])
end
xticks(x_tick_inds)
datetick('x','mmm dd','keepticks','keeplimits')
yticklabels([num2str(yticks') repmat('\%', length(yticks), 1)])
legendStr = {};
if  any(categories==1)
   legendStr = [legendStr; 'Retail \& Recreation']; 
end
if  any(categories==2)
   legendStr = [legendStr; 'Grocery \& Pharmacy']; 
end
if  any(categories==3)
   legendStr = [legendStr; 'Parks']; 
end
if  any(categories==4)
   legendStr = [legendStr; 'Transit stations']; 
end
if  any(categories==5)
   legendStr = [legendStr; 'Workplaces']; 
end

legend(legendStr, 'Interpreter','latex','location','best','FontSize',legendFontSize)
%legend({'First PC', 'Second PC'}, 'Interpreter','latex','location','best','FontSize',tickFontSize)
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, [prefix '' mode '_u.fig'])
exportgraphics(gca, [prefix '' mode '_u.eps'], 'ContentType', contentType)

%% Value of beta
% figure(2); clf; hold on;
% b_baseline = mob_posy(mobility_data(1:T)', c_mob(categories), alpha_mob(categories), b_mob, N, beta_max);
% %plot(N*Mult_u*b,'linewidth',3)
% plot(N*b_baseline,'-r','linewidth',3)
% %plot(N*bhigh*ones(T,1),'--r')
% %plot(N*blow*ones(T,1),'--r')
% %set(gca, 'YScale', 'log')
% set(gca, 'FontSize', tickFontSize);
% title('Value of $N\times\beta(t)$','Interpreter','latex','FontSize',fontSize)
% xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
% ylabel('$N\times\beta(t)$ (log scale)','Interpreter','latex','FontSize',fontSize)
% xlim([1 T-3])
% set(gca,'TickLabelInterpreter','latex')
% savefig(gcf, [prefix '' mode '_beta.fig'])
% exportgraphics(gca, [prefix '' mode '_beta.eps'], 'ContentType', contentType)

%% Evolution of epidemic
if num_mob_categories == 1
    mob_data = mobility_data(1:T)';
else
    mob_data = mobility_data(1:T,:);
end
[D_predicted, H_predicted] = make_trajectory(parameters, [Mult_u*u; zeros(3, num_mob_categories)]);
[D_actual, H_actual] = make_trajectory(parameters, mob_data);
% D_actual_linear = make_trajectory_linear(parameters, mobility_data(1:T)');

% Deaths, including confidence interval
figure(5); clf; hold on;
plot(dates, D_predicted(1:end),'linewidth',3,'marker',Markers{1},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
if strcmp(mode,'Min_Cost')
    plot(dates, [D0; D],'-r','linewidth',3,'marker',Markers{2},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
else
    plot(dates(1:end-1), [D0 D],'-r','linewidth',3,'marker',Markers{2},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
end
if exist('eta_p','var') && exist('eta_n','var')
    D_upper(1) = D0;
    D_lower(1) = D0;
    D_upper(2) = D0 + rHD*H0;
    D_lower(2) = D0 + rHD*H0;
    for t = 3:T
        D_upper(t) = D_upper(t-1) + rHD*(H(t-1) + eta_p(t-1) + eta_n(t-1));
        D_lower(t) = D_lower(t-1) + max(0, rHD*(H(t-1) - eta_p(t-1) - eta_n(t-1)));
    end
    xShade = [dates, fliplr(dates)];
    yShade = [max(D_upper, [D0 D(1:end-1)]), fliplr(min(D_lower, [D0 D(1:end-1)]))];
    figShade = fill(xShade, yShade, 'r');
    figShade.FaceAlpha = 0.15;
    figShade.LineStyle = 'none';
end
plot(dates, D_actual(1:end),'-g','linewidth',3,'marker',Markers{3},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
plot(dates, death_data*N,'--k','linewidth',3)
set(gca, 'FontSize', tickFontSize);
title('Number of cumulative deaths','Interpreter','latex','FontSize',fontSize)
xlabel('Date','Interpreter','latex','FontSize',fontSize)
ylabel('Cumulative deaths','Interpreter','latex','FontSize',fontSize)
xlim([dates(1) dates(end-1)])
xticks(x_tick_inds)
datetick('x','mmm dd','keepticks','keeplimits')
if exist('eta_p','var') && exist('eta_n','var')
    legend('Deaths using optimal $\mathbf{u}^\star(t)$', 'Deaths using $\mathbf{u}^\star(t)$ when $S(t)=S_0$', '$\epsilon$-confidence interval for deaths when $S(t)=S_0$', 'Deaths using baseline $\mathbf{m}(t)$', 'Recorded deaths', 'location', 'best', 'Interpreter','latex','FontSize',tickFontSize)
else
    legend('$\tilde{D}(t)$ using $\mathbf{u}^\star(t)$', '$D(t)$ using $\mathbf{u}^\star(t)$', '$\hat{D}(t)$ using $\mathbf{m}(t)$', 'Recorded deaths', 'location', 'best', 'Interpreter','latex','FontSize',tickFontSize)
end
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, [prefix '' mode '_trajectory_D.fig'])
exportgraphics(gca, [prefix '' mode '_trajectory_D.eps'], 'ContentType', contentType)

% Hospitalized
figure(3); clf; hold on;
plot(dates(1:end), H_predicted(1:end),'linewidth',3,'marker',Markers{1},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
if strcmp(mode,'Min_Cost')
    plot(dates(1:end), [H0; H],'-r','linewidth',3,'marker',Markers{2},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
else
    plot(dates(1:end-1), [H0; H'],'-r','linewidth',3,'marker',Markers{2},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
end
if exist('eta_p','var') && exist('eta_n','var')
    xShade = [1:T, fliplr(1:T)];
    yShade = [H + eta_p + eta_n, fliplr(H - eta_p - eta_n)];
    figShade = fill(xShade, yShade, 'r');
    figShade.FaceAlpha = 0.15;
    figShade.LineStyle = 'none';
end
plot(dates, H_actual(1:end),'-g','linewidth',3,'marker',Markers{3},'MarkerSize',10,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
if strcmp(mode,'Min_Cost')
    plot(dates(1:end-1), tH*ones(size(H)),'--k','linewidth',3)
else
   plot(dates(1:end-2), tH*ones(size(H)),'--k','linewidth',3) 
end
set(gca, 'FontSize', tickFontSize);
title('Number of hospitalized individuals','Interpreter','latex','FontSize',fontSize)
xlabel('Date','Interpreter','latex','FontSize',fontSize)
ylabel('Hospitalized Individuals','Interpreter','latex','FontSize',fontSize)
if exist('eta_p','var') && exist('eta_n','var')
    legend('Hospitalizations using nonlinear model $H(t)$','Hospitalizations using linear model $\overline{H}(t)$','$\epsilon$-confidence interval for $\overline{H}(t)$','Hospitalization threshold $\tau_H$', 'location', 'best', 'Interpreter','latex','FontSize',tickFontSize)
else
    legend('$\tilde{H}(t)$ using $\mathbf{u}^\star(t)$','$H(t)$ using $\mathbf{u}^\star(t)$','$\hat{H}(t)$ using $\mathbf{m}(t)$','$\tau_H$', 'location', 'best', 'Interpreter','latex','FontSize',tickFontSize)
end
ylim([0 tH*1.01])
if strcmp(mode, 'Min_Cost')
    xlim([dates(1) dates(end-1)])
else
   xlim([dates(1) dates(end-2)]) 
end
xticks(x_tick_inds)
datetick('x','mmm dd','keepticks','keeplimits')
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, [prefix '' mode '_trajectory_H.fig'])
exportgraphics(gca, [prefix '' mode '_trajectory_H.eps'], 'ContentType', contentType)

