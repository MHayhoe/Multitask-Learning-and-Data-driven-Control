% Import the learned parameters
initialize_from_python
fontSize = 16;
tickFontSize = 14;
uInc = eps;
uMax = 0.5;

% Get a range of beta values
b = mob_posy((eps:uInc:uMax)', c_mob, alpha_mob, b_mob, N);

% Build the final death values
final_deaths = zeros(size(b));
D_vals = [];

for b_i = 1:size(b,1)
    Mnew = [1 - rEI - rEA, S0*b(b_i), gA*S0*b(b_i);
        rEI, 1 - rIR, 0;
        rEA, 0, 1 - rAR];
    X(:,1) = Mnew*[E0 I0 A0]';
    H(1) = (1 - rHR - rHD)*H0 + rIH*I0 + b(b_i)*0;
    for t=2:T-1
        Mnew = [1 - rEI - rEA, S0*b(b_i), gA*S0*b(b_i);
            rEI, 1 - rIR, 0;
            rEA, 0, 1 - rAR];
        X(:,t) = Mnew*X(:,t-1);
        H(t) = (1 - rHR - rHD)*H(t-1) + rIH*X(2,t-1);
    end
    D_vals(b_i,:) = H * CFR;
    final_deaths(b_i) = sum(H) * CFR;
end

% Input response plot
figure(5); clf;
plot(eps:uInc:uMax, final_deaths,'linewidth',2)
set(gca, 'FontSize', tickFontSize);
title('Input response of the compartmental model','Interpreter','latex','FontSize',fontSize)
xlabel('Control action $u$ (fixed through time)','Interpreter','latex','FontSize',fontSize)
ylabel(['Cumulative deaths after ' num2str(T) ' days'],'Interpreter','latex','FontSize',fontSize)
set(gca,'TickLabelInterpreter','latex')

% Comparison of trajectories
figure(6); clf;
semilogy(round(D_vals(1:100:end,:))','linewidth',2)
set(gca, 'FontSize', tickFontSize);
title('Trajectories for different control actions (fixed through time)','Interpreter','latex','FontSize',fontSize)
xlabel('Time $$t$','Interpreter','latex','FontSize',fontSize)
ylabel('Incident Deaths $D(t)$ (log scale)','Interpreter','latex','FontSize',fontSize)
legend(num2str((eps:0.1:uMax)'),'location','Northwest','Interpreter','latex','FontSize',fontSize)
set(gca,'TickLabelInterpreter','latex')
