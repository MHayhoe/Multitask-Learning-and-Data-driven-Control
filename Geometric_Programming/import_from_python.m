params = readtable('../../Covid-19/posynomial_params.csv','delimiter',',','ReadRowNames',true);
consts = readtable('../../Covid-19/posynomial_consts.csv','delimiter',',','ReadRowNames',true);

% Read in mobility and deaths data
% pyenv('Version', '/Library/Frameworks/Python.framework/Versions/3.8/bin/python3')
directory = '2020-Nov-12-15-12-07';
fid = py.open(['/Users/Mikhail/Dropbox/Covid-19/Plots/' directory '/consts.pickle'],'rb');
data = py.pickle.load(fid);
mobility_data = double(data{'mobility_data'});
death_data = double(data{'death_data'});

fid = py.open(['/Users/Mikhail/Dropbox/Covid-19/Plots/' directory '/opt_params.pickle'],'rb');
param_data = py.pickle.load(fid);

% Number of mobility categories
num_mob_categories = 1; %5;
categories = [1]; %[1,2,3,4,5];
num_categories = max(size(categories));
parameters.categories = categories;

% Populations
n = double(data{'n'});

% Number of regions
num_regions = max(size(n));

% Transition rates
rho_EI = sigmoid(params{startsWith(params.Properties.RowNames,'rho_EI_coeffs'), 1});
rho_EA = sigmoid(params{startsWith(params.Properties.RowNames,'rho_EA_coeffs'), 1});
rho_AR = sigmoid(params{startsWith(params.Properties.RowNames,'rho_AR_coeffs'), 1});
rho_IR = sigmoid(params{startsWith(params.Properties.RowNames,'rho_IR_coeffs'), 1});
rho_IH = sigmoid(params{startsWith(params.Properties.RowNames,'rho_IH_coeffs'), 1});
rho_HR = sigmoid(params{startsWith(params.Properties.RowNames,'rho_HR_coeffs'), 1});

% Initial condition: [E_0, I_0, R_0 & D_0, A_0, H_0]
x_0 = exp(params{startsWith(params.Properties.RowNames,'initial_condition'), :});
x_0 = x_0(~isnan(x_0));
x_0 = reshape(x_0, [num_regions, max(size(x_0))/num_regions]);

% Case fatality ratio
fatality_H = sigmoid(params{startsWith(params.Properties.RowNames,'fatality_H'), 1});

% Terms for posynomial fit of beta as a function of mobility
c_mob = exp(params{startsWith(params.Properties.RowNames,'beta_I_coeffs'), (num_mob_categories+1):(2*num_mob_categories)});
alpha_mob = params{startsWith(params.Properties.RowNames,'beta_I_coeffs'), 1:num_mob_categories};
b_mob = exp(params{startsWith(params.Properties.RowNames,'beta_I_bias'), 1});

% Ratio of infectivity of Asymptomatic (vs Infected)
ratio_A = sigmoid(params{startsWith(params.Properties.RowNames,'ratio_A'), 1});

% Time window
T = consts{'T', 1};