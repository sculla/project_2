import geopy
import dotenv
import pandas as pd
from pyzillow.pyzillow import ZillowWrapper, GetDeepSearchResults, GetUpdatedPropertyDetails

geolocator = geopy.Nominatim(user_agent="homes_test_project")
geopy.geocoders.options.default_timeout = 7


apiKey = dotenv.get_key('.env', 'ZilloapiKey')
df = pd.read_pickle('pickle/with_addr.pickle')
zillow_data = ZillowWrapper(apiKey)


# for i in w_add.Address[:10]:
#     if i[-1]== ' ':
#         print(i)
#     elif type(int(i.split()[-1])) == int:
#         print(i, 'is int')

# location.raw['display_name'].split(',')[-2].strip()


def addr_spl(address):
    """
    checking to see if the address has a zipcode
    :param address: input address from df
    :return: address, zip or address, False
    """
    return ' '.join(x for x in address.split()[:-1]), address.split()[-1]


# 'bathrooms': '3.0',
# 'bedrooms': '4',
# 'year_updated': '2015',
# 'num_rooms': '12',
# 'school_district': 'Federal Way, http://www.fwps.org/',
dic = {}


def zillow_call(address):
    addr, zipcode = addr_spl(address)
    deep_search_response = zillow_data.get_deep_search_results(addr, zipcode)
    result = GetDeepSearchResults(deep_search_response)
    for att in result.attribute_mapping:
        att = str(att)
        if result.get_attr(att) is not None:
            dic[att] = result.get_attr(att)
    zillow_id = result.zillow_id
    updated_property_details_response = zillow_data.get_updated_property_details(zillow_id)
    result = GetUpdatedPropertyDetails(updated_property_details_response)
    for att in result.attribute_mapping:
        att = str(att)
        if result.get_attr(att) is not None:
            dic[att] = result.get_attr(att)
    return dic
