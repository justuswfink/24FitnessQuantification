
### all the packages required somewhere in this notebook are imported here
# if you exectute this cell, you can jump to anything below and the package should load
import pandas as pd           

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


## read od timeseries before and after pre-processing
## file names for export
CSV_DATA_FILE = "./data/warringer2003/machine_ready/growthdata_SD-medium.csv"
CSV_EXPORT_FILE = './data/warringer2003/machine_ready/growthdata_SD-medium_pre-processed.csv'


## read in the trimmed od timeseries
df_trimmed = pd.read_csv(CSV_EXPORT_FILE, header = 0, index_col= [0,1,2,3,4],float_precision='round_trip', na_values='removed')
## read in the original OD timeseries before trimming
df_raw  = pd.read_csv(CSV_DATA_FILE, header = 0, index_col= [0,1,2,3,4],float_precision='round_trip', na_values='removed')


t_full_array = np.array(df_raw.iloc[0].index, dtype = float)
t_min = t_full_array.min()
t_max = t_full_array.max()

def get_excess_od_timeseries(name):

    data_row = df_trimmed.loc[name].dropna()

    od_array = data_row.values
    t_array = np.array(data_row.index, dtype = int)
 

    return t_array, od_array

def get_excess_od_timeseries_before_trim(name):

    data_row = df_raw.loc[name]

    od_array = data_row.values
    t_array = np.array(data_row.index, dtype = int)
 

    return t_array, od_array

def get_well_id(name):
    run_no, plate_no, well_no = name[1:4]
    well_id = str(run_no) + str(plate_no) + str(well_no)
    return well_id

def get_genotype(name):
    return name[0]


def get_finite_diff_timeseries(name):
    t_array, od_array = get_excess_od_timeseries(name)
    deriv_fd = np.gradient(np.log(od_array), t_array  , axis = 0, )
    
    return t_array, deriv_fd



def calc_deriv_fd(t_array, f_array):
    return np.gradient(f_array, t_array  , axis = 0, )

