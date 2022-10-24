% Clear workspace
clear all;

% Case fatality ratio is percentage of recovered cases that die
CFR = exp(-4.046036714);

% Parameters from multitask learning model
gA = exp(-1.6626187310203266);
rEI = exp(-1.243029865);      rEA = exp(-2.671505643);
rIR = exp(-3.688975684);      rIH = exp(-1.949259355);
rAR = exp(-3.478793948);      
rHR = exp(-2.74393683) * (1 - CFR);
rHD = exp(-2.74393683) * CFR;

% Initial conditions (in absolute number of individuals)
N = 1559938;
E0 = exp(-2.1388846261716865);
I0 = exp(-3.978781434);
A0 = exp(-4.102491895);
H0 = exp(-2.1644864797);

% Assorted constants
T = 81;             % Horizon in days (a parameter we will play with...)
tH = 1000;        % Hospital threshold (look in literature!!!)
S0 = N - E0 - I0;   % Total number of susceptible individuals
c = 0.02044;           % Avg daily budget!!! We need to find the min budget to keep hospitals under control
blow = 1e-4/N;        % Minimum allowable value of beta
bhigh = 1/N;        % Maximum allowable value of beta
fixed_days = 1;

% For transforming beta to a T-dimensional vector
num_vars = ceil((T-1)/fixed_days);
Mult_b = zeros(T,num_vars);
for tt = 1:T-1
   Mult_b(tt,ceil(tt/fixed_days)) = 1; 
end