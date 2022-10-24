% Clear workspace
clear all;
fontSize = 16;

% Case fatality ratio is percentage of recovered cases that die
CFR = 0.01;

% Parameters from clinical literature !!!
rEI=0.1;    rIR=1/10;
rEA=0.05;   rIH=rIR*(1/9);    rAR=1/5;
rHR=1/14;   rHD=rHR*(90/10);
gE=1/3;     gA=1.5;

% Initial conditions (in absolute number of individuals)
N=0.5e6;
E0=5; I0=0; A0=0; H0=0;  

%% The Geometric Program

T = 30;             % Horizon in days (a parameter we will play with...)
tH = 1e-3*N;        % Hospital threshold (look in literature!!!)
S0 = N - E0 - I0;   % Total number of susceptible individuals
c = 0.1;           % Avg daily budget!!! We need to find the min budget to keep hospitals under control
blow = 1e-7;        % Minimum allowable value of beta
bhigh = 0.5;        % Maximum allowable value of beta

% Constant values
% Mbar = [1 - rEI - rEA, 0, 0, 0;
%         rEI, 1 - rIR - rIH, 0, 0;
%         rEA, 0, 1 - rAR, 0;
%         0, rIR, 0, 1 - rHR - rHD];

cvx_solver mosek
cvx_begin gp

variables b(T-1)   % careful with indexes

% Build values of H
% Beware indices here: M(t) = M_{t-1}, but H(t) = H_{t}.
P = [1 - rEI, S0*b(1); rEI, 1 - rIR]; % Mbar + [gE*S0*b(1), S0*b(1), gA*S0*b(1), 0; zeros(3,4)];
I(1) = [0 1]*P*[E0 I0]';
for t=2:T
    Mnew = [1 - rEI, S0*b(t-1); rEI, 1 - rIR]; %[gE*S0*b(t), S0*b(t), gA*S0*b(t), 0; zeros(3,4)];
    P = Mnew*P;   % careful with indexes
    I(t) = [0 1]*P*[E0 I0]'; %[0 0 0 1]*P*[E0 I0 A0 H0]';
end

% Create the GP

minimize(sum(I))

%subject to

%I*rIH/tH <= ones(1,T);

(1/(inv(blow)-inv(bhigh)))*sum(b(1:end-1).^(-1)) <= (T-2)*(c+inv(bhigh)/(inv(blow)-inv(bhigh)));

blow*ones(T-1,1) <= b;
b <= bhigh*ones(T-1,1);

cvx_end

disp([num2str(round(sum(I)*rIR*CFR)) ' total deaths.'])

figure(1); clf; hold on;
plot((b(1:end-1).^-1 - bhigh^-1)/(blow^-1 - bhigh^-1))
plot(c*ones(T,1),'--r')
title('$f_{NPI}$','Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
ylabel('Cost','Interpreter','latex','FontSize',fontSize)
xlim([1 T-2])

figure(2); clf; hold on;
plot(b)
plot(bhigh*ones(T,1),'--r')
plot(blow*ones(T,1),'--r')
set(gca, 'YScale', 'log')
title('Value of $\beta(t)$','Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
ylabel('$\beta(t)$ (log scale)','Interpreter','latex','FontSize',fontSize)
xlim([1 T-2])

figure(3); clf; hold on;
plot(I)
title('Number of infected individuals $I(t)$','Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
ylabel('Infected Individuals','Interpreter','latex','FontSize',fontSize)