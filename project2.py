#! /anaconda3/envs/metis/bin/python

from bs4 import BeautifulSoup as bs
from requests import get
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

"""
Notes for the project 2:


98103
425 NW MARKET ST
415 NW MARKET ST
413 NW MARKET ST
411 NW MARKET ST
411 B NW MARKET ST
403 NW MARKET ST
719 NW MARKET ST
709 NW MARKET ST
701 NW MARKET ST
707 NW MARKET ST
	

"""
ZILLOW_API_KEY= 'X1-ZWz1h8esuyxy4r_6sext'
def zillow():
    from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults, GetUpdatedPropertyDetails
    homes = {'98103': [425 NW MARKET ST
415 NW MARKET ST
413 NW MARKET ST
411 NW MARKET ST
411 B NW MARKET ST
403 NW MARKET ST
719 NW MARKET ST
709 NW MARKET ST
701 NW MARKET ST
707 NW MARKET ST]}
    for
        dic = dict()

        zillow_data = ZillowWrapper(ZILLOW_API_KEY)
        deep_search_response = zillow_data.get_deep_search_results(address, zipcode)
        result = GetDeepSearchResults(deep_search_response)

        for att in result.attribute_mapping:
            att = str(att)
            if result.get_attr(att) != None:
                dic[att] = result.get_attr(att)

        zillow_id = result.zillow_id
        updated_property_details_response = zillow_data.get_updated_property_details(zillow_id)
        result = GetUpdatedPropertyDetails(updated_property_details_response)


        for att in result.attribute_mapping:
            att = str(att)
            if result.get_attr(att) != None:
                dic[att] = result.get_attr(att)
    yield dic

def build_df(re):
    df = pd.DataFrame.from_records(dic, index=np.array([1]))
    df.append(pd.DataFrame.from_records(dic, index=np.array([1])))


def get_links():
    parcel_num = [7518500530]
    for parcel in parcel_num:
        url = f'https://blue.kingcounty.com/Assessor/eRealProperty/Detail.aspx?ParcelNbr={parcel}'
        headers = ({'User-Agent':'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36'})
        response = get(url, headers=headers)
        page = response.text
        soup = bs(page, 'lxml')
        for link in soup.find_all('a'):
            print(link['href'])

if __name__ == '__main__':
    zillow()