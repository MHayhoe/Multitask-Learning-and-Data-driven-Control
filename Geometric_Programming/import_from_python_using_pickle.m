% Pick the region
region = 1;

% Read in mobility and deaths data
% pyenv('Version', '/Library/Frameworks/Python.framework/Versions/3.8/bin/python3')
% directory = '2020-Nov-12-18-57-12';
% directory = '2020-Nov-12-15-12-07';     % Good results, 1-diml
% directory = '2020-Nov-13-18-41-35';
% directory = '2020-Nov-16-21-42-30';
% directory = '2020-Nov-19-12-25-47';
% directory = '2020-Dec-20-19-14-14';
% directory = '2020-Dec-21-10-21-48';
% directory = '2020-Dec-21-15-12-38';       % Good results, 2-diml
% directory = '2020-Dec-24-14-02-28';       % Good results, 5-diml
directory = '2021-Mar-16-13-30-59'; % Newer, 5-dim'l
% directory = '2021-Apr-01-22-45-01'; % Newer, 2 PCs, wrong dates
% directory = '2021-Apr-02-18-04-57'; % Newer, 2 PCs, correct dates
fid = py.open(['/Users/Mikhail/Dropbox/Covid-19/Plots/' directory '/consts.pickle'],'rb');
data = py.pickle.load(fid);
mobility_data = double(data{'mobility_data'});
death_data = double(data{'death_data'});
case_data = double(data{'case_data'});

fid = py.open(['/Users/Mikhail/Dropbox/Covid-19/Plots/' directory '/opt_params.pickle'],'rb');
param_data = py.pickle.load(fid);
params = param_data{region-1};

% Number of mobility categories
num_mob_categories = size(mobility_data,3); %5;
categories = 1:num_mob_categories; %[1,2,3,4,5];
num_categories = max(size(categories));
parameters.categories = categories;

% Populations
N_vals = double(data{'n'});
N = N_vals(region);

% Transition rates
rho_EI = sigmoid(double(params{'rho_EI_coeffs'}));
rho_EA = sigmoid(double(params{'rho_EA_coeffs'}));
rho_AR = sigmoid(double(params{'rho_AR_coeffs'}));
rho_IR = sigmoid(double(params{'rho_IR_coeffs'}));
rho_IH = sigmoid(double(params{'rho_IH_coeffs'}));
rho_HR = sigmoid(double(params{'rho_HR_coeffs'}));

% Initial condition: [E_0, I_0, R_0 & D_0, A_0, H_0]
x_0 = exp(double(params{'initial_condition'}));

% Case fatality ratio
fatality_H = sigmoid(double(params{'fatality_H'})) * double(data{'fatality_max'});

% Terms for posynomial fit of beta as a function of mobility
beta_coeffs = squeeze(double(params{'beta_I_coeffs'}));
if size(beta_coeffs,1) == 1
    c_mob = exp(beta_coeffs(2));
    alpha_mob = beta_coeffs(1);
else
    c_mob = exp(beta_coeffs(2,:));
    alpha_mob = beta_coeffs(1,:);
end
b_mob = exp(double(params{'beta_I_bias'}));
beta_max = double(data{'beta_max'});

% Ratio of infectivity of Asymptomatic (vs Infected)
ratio_A = sigmoid(double(params{'ratio_A'}));

% Percentage of cases that are detected
if isfield(struct(params), 'tau')
    tau = sigmoid(double(params{'tau'}));
    use_tau = true;
else
    use_tau = false;
end

% Time window
T = 90; %double(data{'T'});