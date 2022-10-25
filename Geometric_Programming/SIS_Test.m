% Recovering SIS parameters using a maximum likelihood approach

% beta:     Probability of spreading infection (n-by-1)
% delta:    Probability of self-curing (n-by-1)
% steps:    Number of time steps
% Adj:      Adjacency matrix of the network

n = size(Adj,1); % Number of nodes

I = zeros(n,steps); % Probability of being infected
I(:,1) = initial_infection;
 
% Run the simulation
for t = 1:(steps - 1)
    for i = 1:n
       product = Adj(:,i).*(ones(n,1) - beta.*I(:,t));
       I(i,t+1) = I(i,t)*(1 - delta(i)) + (1 - I(i,t))*(1 - prod(product(product ~= 0))); 
    end
end

% Calculate susceptible ratios
S = ones(size(I)) - I;