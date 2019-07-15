from os import path

import numpy as np
import pandas as pd


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

    new_df = pd.read_pickle('data/full_list.pickle')
    new_df.reset_index(inplace=True)
    new_df.drop('index', inplace=True, axis=1)
    new_df.drop_duplicates(['Parcel'], inplace=True)
    conditions = {'Average': 3, 'Very Good': 5, 'Good': 4, 'Fair': 2, 'Poor': 1}

    for col in new_df.columns:
        new_df[col].replace({'': np.nan}, inplace=True)
        new_df[col].replace({' ': np.nan}, inplace=True)
        (new_df[(new_df['Parcel'] != np.nan)])
    for idx_to_int in ['Sale price', 'Adjusted sale price', 'Assessed Value',
                       'Year built / renovated', 'Stories', 'Living units',
                       'Above grade living area', 'Total living area', 'Total basement',
                       'Finished basement', 'Unfinished full', 'Sq ft lot']:
        new_df[idx_to_int] = new_df[idx_to_int].astype('float64')
    new_df.drop(['Picture', 'Excise tax number', 'Sales warning'], axis=1, inplace=True)
    new_df['Year built / renovated'] = new_df['Year built / renovated'].astype('int32')
    new_df['Building grade'] = new_df['Building grade'].astype('str')
    # Paul allan's house was sold.. cant
    new_df = new_df[new_df['Building grade'] != 'Exceptional Properties']
    new_df['Building grade'] = new_df['Building grade'].apply(get_num)
    # dropped mobiles
    new_df = new_df[new_df['Mobile home'] == '\xa0']
    # vacant lots
    new_df = new_df[new_df['Building grade'] > 0]
    new_df['Building condition'] = new_df['Building condition'].replace(conditions)
    new_df['Building Age'] = new_df['Year built / renovated'].apply(lambda x: 2018 - x)
    for col in ['Environmental', 'Nuisances', 'Topography']:
        new_df[col] = new_df[col].apply(is_yes_to_numaric)
    new_df['Percentage complete'] = new_df['Percentage complete'].apply(percent_compl)
    new_df['Below grade living area'] = new_df['Total living area'] - new_df['Above grade living area']

    new_df['BG^2'] = new_df['Building grade'].apply(lambda x: x**2)
    new_df['BC^2'] = new_df['Building condition'].apply(lambda x: x ** 2)

    test_col = ['Sale price', 'Assessed Value', 'BG^2', 'BC^2',
                'Stories', 'Living units', 'Above grade living area', 'Below grade living area',
                'Total basement', 'Finished basement', 'Sq ft lot',
                'Topography', 'Environmental', 'Nuisances', 'Building Age']
    col_names = [x.replace(' ', '_') for x in test_col]
    new_names = dict(zip(test_col, col_names))
    test_df = new_df[test_col]
    test_df = test_df.rename(new_names, axis=1)

    # Dropping $0-100,000 sale price == QUIT CLAIM DEED; RELATED PARTY, FRIEND, OR NEI...
    test_df = test_df[test_df['Sale_price'] > 1e5]
    test_df = test_df[test_df['Assessed_Value'] > 100000]

    test_df = test_df[test_df['Building_Age'] >= 1]
    # final index fixing
    test_df.reset_index(inplace=True)
    test_df.drop(['index'], axis=1, inplace=True)
    test_df.to_pickle('data/.2018_house_data_frame.pickle')
    assert path.exists('data/.2018_house_data_frame.pickle'), 'failed to write pickle'
