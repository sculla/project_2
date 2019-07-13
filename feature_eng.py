import numpy as np
import pandas as pd


homes = pd.read_pickle('.2018_house_data_frame.pickle')


homes['Percentage complete_log'] = homes['Percentage complete'].apply(lambda x: np.log(x)/np.log(100))