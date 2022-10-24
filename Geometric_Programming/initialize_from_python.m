% Clear workspace
clear all;

% Import the data saved in Python
import_from_python_using_pickle

% Case fatality ratio is percentage of recovered cases that die
CFR = fatality_H;

% Parameters from multitask learning model
gA = ratio_A;
rEI = rho_EI;      rEA = rho_EA;
rIR = rho_IR;      rIH = rho_IH;
rAR = rho_AR;      
rHR = rho_HR * (1 - CFR);
rHD = rho_HR * CFR;

% Mobility and death data
start_deaths = double(data{'begin_cases'});
mobility_data = squeeze(mobility_data(region,:,:));
death_data = squeeze(death_data(region,start_deaths:start_deaths+T));
case_data = squeeze(case_data(region,start_deaths:start_deaths+T));

% Initial conditions (in absolute number of individuals)
H0 = N * x_0(5);
E0 = N * x_0(1);
if use_tau
    I0 = case_data(1) * N * tau;
else
    I0 = N * x_0(2);
end
A0 = N * x_0(4);
D0 = death_data(2) * N;
R0 = N * (death_data(2) / double(data{'fatality_max'}) + x_0(3));

% H0 = 227;
% E0 = 30; %floor(H0 * x_0(region, 1) / x_0(region, 5)); 
% I0 = 20; %floor(H0 * x_0(region, 2) / x_0(region, 5)); 
% A0 = 20;
% D0 = 0;

% Assorted constants
tH = 5279*0.5;        % Hospital threshold
S0 = N - E0 - I0 - A0 - H0 - R0 - D0;   % Total number of susceptible individuals
cost_per_category = ones(num_categories, 1);  % Cost of a lost visit per each category
fixed_days = 7;
u_min = 1e-4;        % Minimum value of u
u_max = 1;        % Maximum value of u
discount_factor = 0.99;  % Discount factor for future deaths
infinite_horizon_discount = discount_factor^T/(1 - discount_factor); % Assuming hospitalizations stay the same, compute total discount factor
upper_ratio = 1.05; % limits how much u(t) can grow: u(t+1) <= upper_ratio*u(t)
lower_ratio = 0.9; % limits how much u(t) can shrink: u(t+1) >= lower_ratio*u(t)

% Import the raw mobility data
% region_fips = 42101;
% start_day = 180; % 200 = September 1st, 2020
% num_days = 60 + T; % Until September 21st, 2020
% Table = readtable('../../Covid-19/Data/Global_Mobility_Report.csv');
% inds = Table.census_fips_code == region_fips;
% mob_data_raw = table2array(Table(inds,10:15))';
% mob_data_raw = mob_data_raw(:, start_day:start_day + num_days);

% For the robust GP
uncertain_epsilon = 0.95;
chi_sq_quant = chi2inv(uncertain_epsilon, num_categories);
Sigma = cell(T,1);
for t=1:T
    Sigma{t} = 0.001*diag(std(mobility_data).^2);
    %Sigma{t} = cov(mobility_data);
end

% Construct the parameter structure
parameters.gA = gA;
parameters.rEI = rEI;
parameters.rEA = rEA;
parameters.rIR = rIR;
parameters.rIH = rIH;
parameters.rAR = rAR;
parameters.rHR = rHR;
parameters.rHD = rHD;
parameters.CFR = CFR;
parameters.N = N;
parameters.S0 = S0;
parameters.E0 = E0;
parameters.I0 = I0;
parameters.A0 = A0;
parameters.H0 = H0;
parameters.R0 = R0;
parameters.D0 = D0;
parameters.c_mob = c_mob;
parameters.alpha_mob = alpha_mob;
parameters.b_mob = b_mob;
parameters.beta_max = beta_max;
parameters.T = T;

% For transforming beta to a T-dimensional vector
num_vars = ceil((T)/fixed_days);
Mult_u = zeros(T,num_vars);
for tt = 1:T-1
   Mult_u(tt,ceil(tt/fixed_days)) = 1; 
end