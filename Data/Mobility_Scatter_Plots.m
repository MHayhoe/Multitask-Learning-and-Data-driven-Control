region = 'US-PA';
region_name = 'Pennsylvania';

T = readtable('Global_Mobility_Report.csv');
inds = strcmp(T.iso_3166_2_code, region);
mob_data = table2array(T(inds,8:13))';
z = zeros(1,size(mob_data,2));

cmap = jet(size(mob_data,2));
colormap jet
figure(1);

for ii = 1:6
    for jj = 1:6
        if ii == jj
            subplot(6,6,(ii-1)*6 + ii);
            histogram(mob_data(ii,:),30)
        else
            subplot(6,6,(ii-1)*6+jj);
            hold on
            x = mob_data(ii,:);
            y = mob_data(jj,:);
            col = 1:size(mob_data,2);
            % surface([x;x],[y;y],[z;z],[col;col],'edgecol','interp','linew',2)
            scatter(x, y, 10, cmap, 'filled')
            hold off
        end
    end
end
sgtitle([region_name, ' Mobility Patterns'])

first_diff = mob_data(:,2:end)' - mob_data(:,1:end-1)';
Covariance = cov(first_diff);
Correlation = corrcoef(first_diff);
[V,L] = eigs(Covariance);
L = diag(L);
explained_variance = cumsum(L)/sum(L);
W = V(:,1:2);

figure(2); clf; hold on;
X = [W'*mob_data(:,1), W'*mob_data(:,1) + cumsum(W'*first_diff')];
scatter(X(1,:), X(2,:), 25, cmap, 'filled')
hold off;
title([region_name ' mobility in 2 PCs, describing ' num2str(sum(L(1:2))/sum(L)*100) '% of variance'])
