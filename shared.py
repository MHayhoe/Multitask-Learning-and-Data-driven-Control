true_params = {}
consts = {}
# Whether we want to use real data (from NYT) or simulated data
real_data = True

# First day to begin with from the real data (NYT). 0 = Jan 21, 2020
# Set to 25 to begin with mobility data.
# Current up to May 19, 2020
# First day to begin with mobility data (Google). 0 = Feb 15, 2020
# Current up to May 16, 2020
begin = {}

# For plotting in optimization callback function
plot_values = []

###### NOT CURRENTLY IN USE
# Highest rate for transition probabilities, i.e., enforce \rho to be in [0,max_rate]
max_rate = 0.25
