% Load data
initialize_from_python

% Set parameters
fontSize = 24;
tickFontSize = 20;
legendFontSize = 14;
contentType = 'image';
prefix = 'Plots/';
num_regions = length(data{'county_names'});
date_offset = datenum('23-Jan-2020');
dates = start_deaths+date_offset:start_deaths+date_offset+size(mobility_data,1);


region_name = data{'county_names'}(region);
region_name = string(region_name{1}(4:end)) + ', ' + string(region_name{1}(1:2));
region_deaths = double(data{'death_data'});
T_train = double(data{'T'});
T_test = 30;
T_total = T_test + T_train;

recorded_deaths = N_vals(region)*squeeze(region_deaths(region,start_deaths:start_deaths+T_total));

python_params = parameters;
python_params.T = T_total;
python_params.D0 = recorded_deaths(1);
predicted_deaths = make_trajectory(python_params, mobility_data);

% For plot legend
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

% Cumulative Deaths
figure(1); clf; hold on;
plot(predicted_deaths,'linewidth',3)
plot(recorded_deaths,'--k','linewidth',3)
ylims = ylim;
figShade = fill([T_train, T_total, T_total, T_train], [ylims(1), ylims(1), ylims(2), ylims(2)], 'r');
figShade.FaceAlpha = 0.15;
figShade.LineStyle = 'none';
ylim(ylims)
set(gca, 'FontSize', tickFontSize);
xlim([1 T_total])
title(region_name,'Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$ (days)','Interpreter','latex','FontSize',fontSize)
ylabel('Cumulative deaths','Interpreter','latex','FontSize',fontSize)
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, ['Plots/region_' num2str(region) '_predictions.fig'])
exportgraphics(gca, ['Plots/region_' num2str(region) '_predictions.eps'], 'ContentType', contentType)

% Mobility data
figure(2); clf; hold on;
Markers = {'o', '*', 'x', '^', 'diamond'};
for m_ind = 1:size(mobility_data,2)
    plot(dates(1:size(mobility_data,1)),mobility_data(:,m_ind)*100,'linewidth',2,'marker',Markers{m_ind},'MarkerSize',8,'MarkerIndices',[m_ind:7:size(mobility_data,1)])
end
plot(dates(1:size(mobility_data,1)),ones(size(mobility_data,1))*100,'--k','linewidth',2)
ylims = ylim;
ylim(ylims)
set(gca, 'FontSize', tickFontSize);
xlim([1 T_total])
title(region_name,'Interpreter','latex','FontSize',fontSize)
ylabel('Percentage of baseline','Interpreter','latex','FontSize',fontSize)
xlabel('Date','Interpreter','latex','FontSize',fontSize)
xlim([dates(1) dates(end)])
datetick('x','mmm dd','keepticks','keeplimits')
legend(legendStr, 'Interpreter','latex','location','best','FontSize',legendFontSize)
yticklabels([num2str(yticks') repmat('\%', length(yticks), 1)])
set(gca,'TickLabelInterpreter','latex')
savefig(gcf, ['Plots/region_' num2str(region) '_predictions.fig'])
exportgraphics(gca, ['Plots/region_' num2str(region) '_mobility.eps'], 'ContentType', contentType)
    
    % Incident Deaths
%     figure(region + num_regions); clf; hold on;
%     diff_predicted = diff(predicted_deaths);
%     diff_recorded = movmean(diff(region_deaths(region,:)), [6,0]);
%     diff_recorded = N*squeeze(diff_recorded(start_deaths:start_deaths+T_total));
%     plot(diff_predicted,'linewidth',3)
%     plot(diff_recorded,'--k','linewidth',3)
%     ylims = ylim;
%     figShade = fill([T_train, T_total, T_total, T_train], [ylims(1), ylims(1), ylims(2), ylims(2)], 'r');
%     figShade.FaceAlpha = 0.15;
%     figShade.LineStyle = 'none';
%     ylim(ylims)
%     set(gca, 'FontSize', tickFontSize);
%     xlim([1 T_total])
%     title(region_name,'Interpreter','latex','FontSize',fontSize)
%     xlabel('Time $t$ (days)','Interpreter','latex','FontSize',fontSize)
%     ylabel('Incident deaths','Interpreter','latex','FontSize',fontSize)
%     set(gca,'TickLabelInterpreter','latex')
%     savefig(gcf, ['Plots/region_' num2str(region) '_predictions_incident.fig'])
%     exportgraphics(gca, ['Plots/region_' num2str(region) '_predictions_incident.eps'], 'ContentType', contentType)
