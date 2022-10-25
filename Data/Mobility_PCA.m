F = readmatrix('county_list.csv');
cumulative_explained_variance = zeros(10);
i = 1;
names = textscan(fopen('county_list.csv'), '%s %s');
names = names{1};

for f = F(:,2)'
    X = readmatrix(['time_series_vsq_categories-' num2str(f) '.csv']);
    X = X(2:2:end,3:end);
    [~, ~, L] = pca(X);
    cumulative_explained_variance(:,i) = (cumsum(L)./sum(L));
    i = i+1;
end
plot(cumulative_explained_variance,'linewidth',2)
legend(names{2:end},'location','Southeast')
min_index = find(min(cumulative_explained_variance,[],2) > 0.9, 1);