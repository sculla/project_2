import pandas as pd
import numpy as np

from os import path

# %config InlineBackend.figure_format = 'svg'
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 25)
# pd.set_option('display.precision', 3)

def get_num(val):
    if val.lower == 'nan':
        return np.nan
    return float(val.split()[0])

def is_yes_to_numaric(val):
    val = str(val)
    if val == '\xa0':
        return 0
    if val.lower() == 'yes':
        return 1
    return 0

def percent_compl(val):
    val = int(val)
    if val == 0:
        return 100
    return val

if __name__ == '__main__':

    new_df = pd.read_pickle('full_list.pickle')
    new_df.reset_index(inplace=True)
    new_df.drop('index',inplace=True, axis=1)
    new_df.drop_duplicates(['Parcel'], inplace=True)
    conditions = {'Average':3, 'Very Good':5, 'Good':4,'Fair':2, 'Poor':1}

    for col in new_df.columns:
        new_df[col].replace({'': np.nan}, inplace=True)
        new_df[col].replace({' ': np.nan}, inplace=True)
        (new_df[(new_df['Parcel'] != np.nan)])
    for idx_to_int in ['Sale price','Adjusted sale price','Assessed Value',\
              'Year built / renovated','Stories','Living units',\
              'Above grade living area','Total living area','Total basement',\
              'Finished basement','Unfinished full','Sq ft lot']:
        new_df[idx_to_int] = new_df[idx_to_int].astype('float64')
    new_df.drop(['Picture','Excise tax number', 'Sales warning'], axis=1, inplace=True)
    new_df['Year built / renovated'] = new_df['Year built / renovated'].astype('int32')
    new_df['Building grade'] = new_df['Building grade'].astype('str')
    #Paul allan's house was sold.. cant 
    new_df = new_df[new_df['Building grade'] != 'Exceptional Properties']
    new_df['Building grade'] = new_df['Building grade'].apply(get_num)
    #dropped mobiles
    new_df = new_df[new_df['Mobile home'] == '\xa0']
    #vacant lots
    new_df = new_df[new_df['Building grade'] > 0]
    new_df['Building condition'] = new_df['Building condition'].replace(conditions)
    new_df['Building Age'] = new_df['Year built / renovated'].apply(lambda x: 2018 - x)
    for col in ['Environmental','Nuisances','Topography']:
       new_df[col] = new_df[col].apply(is_yes_to_numaric)
    new_df['Percentage complete'] = new_df['Percentage complete'].apply(percent_compl)
    test_df = new_df[['Sale price',
           'Assessed Value',
           'Building grade', 'Building condition',
           'Stories', 'Living units',
           'Above grade living area', 'Total living area', 'Total basement',
           'Finished basement', 'Percentage complete',
           'Sq ft lot',
           'Topography', 'Environmental', 'Nuisances',
           'Building Age']]
    test_df.to_pickle('2018_house_data_frame.pickle')
    assert path.exists('2018_house_data_frame.pickle'), 'failed to write pickle'
