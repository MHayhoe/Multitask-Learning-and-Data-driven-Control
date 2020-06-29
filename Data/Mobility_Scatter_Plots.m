region = 'US-NY';
region_name = 'New York';

T = readtable('Global_Mobility_Report.csv');
inds = strcmp(T.iso_3166_2_code, region);
mob_data = table2array(T(inds,8:13))';

cmap = jet(size(mob_data,2));

for ii = 1:6
    for jj = 1:6
        if ii == jj
            subplot(6,6,(ii-1)*6 + ii);
            histogram(mob_data(ii,:),30)
        else
            subplot(6,6,(ii-1)*6+jj);
            %scatter(mob_data(ii,:), mob_data(jj,:), 10, cmap, 'filled')
            plot(mob_data(ii,:), mob_data(jj,:), 'LineColor', cmap)
        end
    end
end
sgtitle([region_name, ' Mobility Patterns'])
