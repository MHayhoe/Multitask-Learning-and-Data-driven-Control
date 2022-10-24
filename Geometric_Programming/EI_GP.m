% Clear workspace
clear all;
fontSize = 16;

% Case fatality ratio is percentage of recovered cases that die
CFR = 0.01;

% Parameters from clinical literature !!!
rEI=0.1;    rIR=1/10;

% Initial conditions (in absolute number of individuals)
N=0.5e6;
E0=100;
I0=0;

% Assorted constants
T = 10;             % Horizon in days (a parameter we will play with...)
tH = 1e-3*N;        % Hospital threshold (look in literature!!!)
tI = 70;
S0 = N - E0 - I0;   % Total number of susceptible individuals
c = 0.5;           % Avg daily budget!!! We need to find the min budget to keep hospitals under control
blow = 1e-9;        % Minimum allowable value of beta
bhigh = 0.5;        % Maximum allowable value of beta

%% The Geometric Program
cvx_solver mosek
cvx_begin gp

variables b(T-1)   % careful with indexes

% Build values of I
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
P = [1 - rEI, S0*b(1); rEI, 1 - rIR];
I(1) = [0 1]*P*[E0 I0]';
for t=2:T
    Mnew = [1 - rEI, S0*b(t-1); rEI, 1 - rIR];
    P = Mnew*P;
    I(t) = [0 1]*P*[E0 I0]';
end

% Create the GP
minimize(sum(I(1:end-1)))

subject to

I <= tI*ones(1,T);
(1/(inv(blow)-inv(bhigh)))*sum(b(1:end-1).^(-1)) <= (T-2)*(c+inv(bhigh)/(inv(blow)-inv(bhigh)));
blow*ones(T-1,1) <= b;
b <= bhigh*ones(T-1,1);

cvx_end

%% Plot results
do_plots