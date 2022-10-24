function [D_full, H_full] = make_trajectory(p, u_values)
    % Get baseline control
    % u_values = mobility_data(1:T,categories);
    % u_baseline = [Mult_u*u; zeros(3, num_mob_categories)];
    b_baseline = mob_posy(u_values, p.c_mob(p.categories), p.alpha_mob(p.categories), p.b_mob, p.N, p.beta_max);

    % Evolve the dynamics
    S_full(1) = p.S0;
    E_full(1) = p.E0;
    I_full(1) = p.I0;
    A_full(1) = p.A0;
    H_full(1) = p.H0;
    R_full(1) = p.R0;
    D_full(1) = p.D0;
    for t = 1:p.T
        S_full(t+1) = p.N - E_full(t) - I_full(t) - A_full(t) - H_full(t) - R_full(t) - D_full(t);
        E_full(t+1) = (1 -  p.rEI - p.rEA)*E_full(t) + S_full(t)*b_baseline(t)*(p.gA*A_full(t) + I_full(t));
        I_full(t+1) = (1 - p.rIR - p.rIH)*I_full(t) + p.rEI*E_full(t);
        A_full(t+1) = (1 - p.rAR)*A_full(t) + p.rEA*E_full(t);
        H_full(t+1) = (1 - p.rHR - p.rHD)*H_full(t) + p.rIH*I_full(t);
        R_full(t+1) = R_full(t) + p.rIR*I_full(t) + p.rAR*A_full(t) + p.rHR*H_full(t);
        D_full(t+1) = D_full(t) + p.rHD*H_full(t);
    end
%     figure; hold on;
%     plot(E_full,'linewidth',3)
%     plot(I_full,'linewidth',3)
%     plot(A_full,'linewidth',3)
%     plot(H_full,'linewidth',3)
%     %plot(R_full,'linewidth',3)
%     plot(D_full,'linewidth',3)
%     legend('E','I','A','H','D','location','northwest')