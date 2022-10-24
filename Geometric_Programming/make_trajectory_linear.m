function D = make_trajectory_linear(p, u_values)
    b_baseline = mob_posy(u_values, p.c_mob(p.categories), p.alpha_mob(p.categories), p.b_mob, p.N, p.beta_max);
    Mnew = [1 - p.rEI - p.rEA, p.S0*b_baseline(1), p.gA*p.S0*b_baseline(1);
        p.rEI, 1 - p.rIR, 0;
        p.rEA, 0, 1 - p.rAR];
    X(:,1) = Mnew*[p.E0 p.I0 p.A0]';
    H(1) = (1 - p.rHR - p.rHD)*p.H0 + p.rIH*p.I0;
    D(1) = p.D0;
    for t=2:p.T-1
        Mnew = [1 - p.rEI - p.rEA, p.S0*b_baseline(t), p.gA*p.S0*b_baseline(t);
            p.rEI, 1 - p.rIR, 0;
            p.rEA, 0, 1 - p.rAR];
        X(:,t) = Mnew*X(:,t-1);
        H(t) = (1 - p.rHR - p.rHD)*H(t-1) + p.rIH*X(2,t-1);
        D(t) = D(t-1) + p.rHD*H(t-1);
    end