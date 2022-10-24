% Set parameters
fontSize = 16;

%% Cost
figure(1); clf; hold on;
plot(Mult_u*(b.^-1 - bhigh^-1)/(blow^-1 - bhigh^-1))
plot(daily_budget*ones(T,1),'--r')
title('$f_{NPI} = \left(\beta(t)^{-1} - \overline{\beta}^{-1}\right)/\left(\underline{\beta}^{-1} - \overline{\beta}^{-1}\right)$','Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
ylabel('Cost','Interpreter','latex','FontSize',fontSize)
xlim([1 T-3])

%% Value of beta
figure(2); clf; hold on;
plot(N*Mult_u*b)
plot(N*bhigh*ones(T,1),'--r')
plot(N*blow*ones(T,1),'--r')
set(gca, 'YScale', 'log')
title('Value of $N\times\beta(t)$','Interpreter','latex','FontSize',fontSize)
xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
ylabel('$N\times\beta(t)$ (log scale)','Interpreter','latex','FontSize',fontSize)
xlim([1 T-3])

%% Evolution of epidemic
if exist('H','var')
    disp([num2str(round(sum(H)*rHD)) ' total deaths.'])
    
    figure(3); clf; hold on;
    plot(H)
    plot(tH*ones(size(H)),'--r')
    title('Number of hospitalized individuals $H(t)$','Interpreter','latex','FontSize',fontSize)
    xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
    ylabel('Hospitalized Individuals','Interpreter','latex','FontSize',fontSize)
    xlim([1 T-1])
elseif exist('I','var')
    disp([num2str(round(sum(I)*rIR*CFR)) ' total deaths.'])

    figure(3); clf; hold on;
    plot(I)
    title('Number of infected individuals $I(t)$','Interpreter','latex','FontSize',fontSize)
    xlabel('Time $t$','Interpreter','latex','FontSize',fontSize)
    ylabel('Infected Individuals','Interpreter','latex','FontSize',fontSize)
    xlim([1 T-1])
end