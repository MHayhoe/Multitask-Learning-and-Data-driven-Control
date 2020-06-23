import pandas as pd
import numpy as np
import pickle
from astropy.convolution import convolve, Gaussian1DKernel


def import_population_data(filename):
    # Read in the data
    df = pd.read_csv(filename, encoding="ISO-8859-1")

    # Apply county names, and keep only 2019 population estimate
    df['county'] = df['STNAME'] + '-' + df['CTYNAME']
    df = df.filter(items=['county', 'POPESTIMATE2019'])
    counties = df.county.unique()

    population_data = {}

    # Save populations for each county
    for c in counties:
        this_county = df['county'] == c
        population_data[c] = df[this_county].POPESTIMATE2019.values[0]

    return population_data


# Imports data from Google's Global Mobility Report. Data starts on Feb 15, 2020.
def import_mobility_data(filename, country='US'):
    # Read in the data
    df = pd.read_csv(filename, encoding="ISO-8859-1", low_memory=False)

    # Only keep data for the required country
    country_filter = df['country_region_code'] == country
    df = df[country_filter]

    # To transform states to abbreviations
    us_state_abbrev = {
        'Alabama': 'AL',
        'Alaska': 'AK',
        'American Samoa': 'AS',
        'Arizona': 'AZ',
        'Arkansas': 'AR',
        'California': 'CA',
        'Colorado': 'CO',
        'Connecticut': 'CT',
        'Delaware': 'DE',
        'District of Columbia': 'DC',
        'Florida': 'FL',
        'Georgia': 'GA',
        'Guam': 'GU',
        'Hawaii': 'HI',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Iowa': 'IA',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Maine': 'ME',
        'Maryland': 'MD',
        'Massachusetts': 'MA',
        'Michigan': 'MI',
        'Minnesota': 'MN',
        'Mississippi': 'MS',
        'Missouri': 'MO',
        'Montana': 'MT',
        'Nebraska': 'NE',
        'Nevada': 'NV',
        'New Hampshire': 'NH',
        'New Jersey': 'NJ',
        'New Mexico': 'NM',
        'New York': 'NY',
        'North Carolina': 'NC',
        'North Dakota': 'ND',
        'Northern Mariana Islands': 'MP',
        'Ohio': 'OH',
        'Oklahoma': 'OK',
        'Oregon': 'OR',
        'Pennsylvania': 'PA',
        'Puerto Rico': 'PR',
        'Rhode Island': 'RI',
        'South Carolina': 'SC',
        'South Dakota': 'SD',
        'Tennessee': 'TN',
        'Texas': 'TX',
        'Utah': 'UT',
        'Vermont': 'VT',
        'Virgin Islands': 'VI',
        'Virginia': 'VA',
        'Washington': 'WA',
        'West Virginia': 'WV',
        'Wisconsin': 'WI',
        'Wyoming': 'WY',
    }

    # Combine State and County information
    national_level = pd.isna(df['sub_region_1'])        # This is at the national level
    has_state = [not x for x in national_level]         # These have state affiliations
    no_county = pd.isna(df['sub_region_2'])             # Has no county affiliation
    state_level = np.logical_and(no_county, has_state)  # At the state level (no county, not national)
    county_level = [not x for x in no_county]
    # Turn state names into abbreviations
    df['sub_region_1'][has_state] = df['sub_region_1'][has_state].apply(lambda x: us_state_abbrev[x])

    # For national, use the country name
    df.loc[national_level, 'region'] = country
    # For states, just use state name
    df.loc[state_level, 'region'] = df['sub_region_1']
    # For counties, concatenate state and county name
    df.loc[county_level, 'region'] = df['sub_region_1'] + '-' + df['sub_region_2']
    df.drop(['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2'],axis=1)

    # Only keep data points with proper county, i.e., not country-level
    # has_county = pd.isna(df['county'])
    # has_county = [not x for x in has_county]
    # df = df[has_county]

    # Find all unique counties and dates
    regions = df.region.unique()
    dates = df.date.unique()

    # For interpolating between nan values
    kernel_size = len(dates)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = Gaussian1DKernel(2,x_size=kernel_size)

    data = {}

    for c in regions:
        # Make a data frame just for this county
        df_region = df[df['region'] == c]

        # Find missing dates and add NaNs
        region_dates = df_region.date.unique()
        missing_dates = np.setdiff1d(dates, region_dates)
        new_data = [{'region': c, 'date': d} for d in missing_dates]
        df_region = df_region.append(new_data,ignore_index=True)
        df_region.sort_values(by='date')

        df_region = df_region.filter(items=['retail_and_recreation_percent_change_from_baseline',
                                            'grocery_and_pharmacy_percent_change_from_baseline',
                                            'parks_percent_change_from_baseline',
                                            'transit_stations_percent_change_from_baseline',
                                            'workplaces_percent_change_from_baseline',
                                            'residential_percent_change_from_baseline'])
        mob_array = df_region.to_numpy()

        # Smooth the data with a Gaussian kernel - deals with NaN values
        for cat_ind in range(6):
            mob_array[:,cat_ind] = convolve(mob_array[:,cat_ind],kernel)

        data[c] = mob_array

    return data


# Imports population and mobility data, removing any entries that aren't in both
def import_data(mobility_name, country='US'):
    land_data, age_data, death_data, case_data = import_safegraph_data()
    print('Land Area, Age, Deaths, and Case Counts data imported.')
    mob_data = import_mobility_data(mobility_name, country)
    print('Mobility data imported.')

    # Remove any entries that aren't in all dictionaries
    # Note that land_data and pop_data have identical key sets
    mob_keys = set(mob_data.keys())
    sg_keys = set(age_data.keys())
    regions_keys = sg_keys & mob_keys
    extra_sg = sg_keys.difference(regions_keys)
    extra_mob = mob_keys.difference(regions_keys)

    for k in extra_sg:
        del land_data[k]
        del age_data[k]
        del death_data[k]
        del case_data[k]

    for k in extra_mob:
        del mob_data[k]

    print('Dictionaries intersected.')

    return land_data, age_data, death_data, case_data, mob_data


# Imports safegraph data at the Census Block Group (CBG) level and aggregates at the county level
# Also imports NYT data on case and death counts, starting from January 21, 2020.
def import_safegraph_data():
    # Read in the data, and set leading zeros
    df = pd.read_csv('Data/cbg_fips_codes.csv', encoding="ISO-8859-1", low_memory=False)
    df['fips'] = df['state_fips'].apply('{0:0>2}'.format) + df['county_fips'].apply('{0:0>3}'.format)

    # Get the regions
    df['region'] = df['state'] + '-' + df['county']

    # Add rows for the states and the national level
    states = df.state.unique()
    df_state = pd.DataFrame(np.hstack((states,'US')), columns=['region'])
    # Set national FIPS code
    df_state.loc[df_state['region'] == 'US', 'fips'] = ''
    # Set state FIPS codes
    last_ind = 0
    for s in states:
        for i in range(last_ind,len(df)):
            if df.state.iloc[i] == s:
                last_ind = i
                df_state.loc[df_state['region'] == s, 'fips'] = df.fips.iloc[i][0:2]
                break

    # Append the state and national rows to the dataframe, and drop unnecessary columns
    df = df.append(df_state,ignore_index=True).filter(items=['region','fips'])

    # Land area data
    df_land = pd.read_csv('Data/cbg_geographic_data.csv', encoding="ISO-8859-1", low_memory=False)
    df_land['census_block_group'] = df_land['census_block_group'].apply('{0:0>12}'.format)
    land_data = {}

    # Male by age:   B01001e2 - B01001e25   (first is total by gender)
    age_fields = ['B01001e' + str(i) for i in range(3,26)]
    # Female by age: B01001e26 - B01001e49  (first is total by gender)
    age_fields += ['B01001e' + str(i) for i in range(27,50)]
    # Age distribution: <5, 5-9, 10-14, 15-17, 18-19, 20, 21, 22-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59,
    #                   60-61, 62-64, 65-66, 67-69, 70-74, 75-79, 80-84, >85.

    # Age data
    df_age = pd.read_csv('Data/cbg_b01.csv', encoding="ISO-8859-1", low_memory=False)
    df_age['census_block_group'] = df_age['census_block_group'].apply('{0:0>12}'.format)
    df_age = df_age.filter(items=['census_block_group']+age_fields)
    age_data = {}

    # Death and case counts (infected people) data
    df_nyt = pd.read_csv('Data/nytimes_infections.csv', encoding="ISO-8859-1", low_memory=False)
    df_nyt['countyFIPS'] = df_nyt['countyFIPS'].apply('{0:0>5}'.format)
    num_nyt_dates = int((len(df_nyt.columns) - 1) / 2)
    df_cases = df_nyt.iloc[:,:num_nyt_dates+1]
    df_deaths = df_nyt.iloc[:, [0] + list(range(num_nyt_dates+1,2*num_nyt_dates+1))]
    death_data = {}
    case_data = {}

    # Aggregate the land area, age distribution data, deaths, and case counts by region
    for index, row in df.iterrows():
        region = row['region']
        fips = row['fips']

        # Land area data - aggregate CBG to region
        filt = df_land['census_block_group'].str.startswith(fips)
        df_fips = df_land[filt]
        area = df_fips['amount_land'].sum() / 1e6  # Convert m^2 to km^2
        land_data[region] = area

        # Age data - aggregate CBG to region
        filt = df_age['census_block_group'].str.startswith(fips)
        df_fips = df_age[filt].drop('census_block_group',axis=1)
        ages = np.sum(df_fips.to_numpy(), axis=1)
        age_data[region] = ages

        # Deaths and case counts
        if len(fips) == 5:
            cases = df_cases[df_cases['countyFIPS'] == fips].to_numpy().T
            deaths = df_deaths[df_deaths['countyFIPS'] == fips].to_numpy().T
        else:
            cases = np.sum(df_cases[df_cases['countyFIPS'].str.startswith(fips)].to_numpy().T, axis=1)
            deaths = np.sum(df_deaths[df_deaths['countyFIPS'].str.startswith(fips)].to_numpy().T, axis=1)
        case_data[region] = cases[1:]
        death_data[region] = deaths[1:]

    return land_data, age_data, death_data, case_data


if __name__ == '__main__':
    # Import and save the data
    land_area_data, age_distribution_data, deaths_data, case_count_data, mobility_data = import_data('Data/Global_Mobility_Report.csv')

    with open('Mobility_US.pickle', 'wb') as handle:
        pickle.dump(mobility_data, handle, protocol=4)

    with open('Age_Distribution_US.pickle', 'wb') as handle:
        pickle.dump(age_distribution_data, handle, protocol=4)

    with open('Land_Area_US.pickle', 'wb') as handle:
        pickle.dump(land_area_data, handle, protocol=4)

    with open('Deaths_US.pickle', 'wb') as handle:
        pickle.dump(deaths_data, handle, protocol=4)

    with open('Case_Counts_US.pickle', 'wb') as handle:
        pickle.dump(case_count_data, handle, protocol=4)
