Awall = rand(1);
Afloor = rand(1);
alpha = 0.1;
beta = 1;
gamma = 2;
delta = 4;

cvx_begin gp
    variables w h d
    maximize( w * h * d )
    subject to
        2*(h*w+h*d) <= Awall;
        w*d <= Afloor;
        alpha <= h/w >= beta;
        gamma <= d/w <= delta;
cvx_end