% Clear workspace
clear all;
fontSize = 16;

% Case fatality ratio is percentage of recovered cases that die
CFR = 0.01;

% Parameters from clinical literature !!!
rEI = 1/5;      rEA = rEI*0.3;
rIR = 1/9;      rIH = 0.1;
rAR = 1/9;      gA = 1.5;
rHR = 1/14;     rHD = rHR*(1/9);

% Initial conditions (in absolute number of individuals)
N = 0.5e6;
E0 = 100;
I0 = 1;
A0 = 1;
H0 = 1;

% Assorted constants
T = 100;             % Horizon in days (a parameter we will play with...)
tH = 18;        % Hospital threshold (look in literature!!!)
tI = 200;
S0 = N - E0 - I0 - A0 - H0;   % Total number of susceptible individuals
c = 0.1;           % Avg daily budget!!! We need to find the min budget to keep hospitals under control
blow = 1e-3;        % Minimum allowable value of beta
bhigh = 0.01;        % Maximum allowable value of beta

%% The Geometric Program
cvx_solver mosek
cvx_begin gp

variables b(T-1)   % careful with indexes

% Build values of H
% Beware indices here: b(t) = \beta(t-1) so M(t) = M_{t-1}, but H(t) = H_{t}.
Mnew = [1 - rEI - rEA, S0*b(1), gA*S0*b(1), 0;
        rEI, 1 - rIR - rIH, 0, 0;
        rEA, 0, 1 - rAR, 0;
        0, rIH, 0, 1 - rHR - rHD];
X(:,1) = Mnew*[E0 I0 A0 H0]';
for t=2:T-1
    Mnew = [1 - rEI - rEA, S0*b(t), gA*S0*b(t), 0;
            rEI, 1 - rIR - rIH, 0, 0;
            rEA, 0, 1 - rAR, 0;
            0, rIH, 0, 1 - rHR - rHD];
    X(:,t) = Mnew*X(:,t-1);
end

% Create the GP
minimize(sum([0 1 0 0]*X))

subject to

%X*[0 0 0 1]' <= tH;
(1/(inv(blow)-inv(bhigh)))*sum(b(1:T-3).^(-1)) <= (T-2)*(c+inv(bhigh)/(inv(blow)-inv(bhigh)));
blow <= b;
b <= bhigh;

cvx_end

%% Plot results
I = [0 1 0 0]*X;
do_plots