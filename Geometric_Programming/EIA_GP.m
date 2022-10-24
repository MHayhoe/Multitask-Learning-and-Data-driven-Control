% Clear workspace
clear all;
fontSize = 16;

% Case fatality ratio is percentage of recovered cases that die
CFR = 0.01;

% Parameters from clinical literature !!!
rEI=0.1;    rIR=1/10;
rEA=0.05;   rAR=1/5;
gE=1/3;     gA=1.5;

% Initial conditions (in absolute number of individuals)
N = 0.5e6;
E0 = 100;
I0 = 0;
A0 = 0;

% Assorted constants
T = 10;             % Horizon in days (a parameter we will play with...)
tH = 1e-3*N;        % Hospital threshold (look in literature!!!)
tI = 1000;
S0 = N - E0 - I0;   % Total number of susceptible individuals
c = 0.1;           % Avg daily budget!!! We need to find the min budget to keep hospitals under control
blow = 1e-7;        % Minimum allowable value of beta
bhigh = 0.5;        % Maximum allowable value of beta

%% The Geometric Program
cvx_solver mosek
cvx_begin gp

variables b(T-1)   % careful with indexes

% Build values of I
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
Mnew = [1 - rEI - rEA, S0*b(1), gA*S0*b(1);
    rEI, 1 - rIR, 0;
    rEA, 0, 1 - rAR];
X(:,1) = Mnew*[E0 I0 A0]';
for t=2:T-1
    Mnew = [1 - rEI - rEA, S0*b(t), gA*S0*b(t);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
    X(:,t) = Mnew*X(:,t-1);
end

% Create the GP
I = [0 1 0]*X;
minimize(sum(I))

subject to

%I <= tI;
(1/(inv(blow)-inv(bhigh)))*sum(b(1:T-1).^(-1)) <= (T-2)*(c+inv(bhigh)/(inv(blow)-inv(bhigh)));
blow <= b;
b <= bhigh;

cvx_end

%% Plot results
do_plots