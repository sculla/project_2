
# Predicting Home Values with LinReg

## Project #2 Proposal Aaron Scull

---

### Scope:  

Buying a home is the largest part of someone's life. It can be difficult to find a
house that is reasonably priced and within the features that you would like. Pre-Sale
Realitors can confidently negotiate the best price for the home for their clients, and 
homeowners this would be able to feel confidant in their purchase by confirming the 
price they paid is at market value. 

### Methodology:

1. Scrape addresses of Seattle homes sold in 2018 from King County Assessors
2. Add Feature - Median Income for Seattle by Zip
3. Pipe the homes through the Zillow API to pull details on parcel
4. Build Linear Reg using 80% of the data set training the numerical features of the set
to predict the sales price


#### Potential additional step:

1. Scrape the Walk Scores for the homes
2. Rebuild LinReg
3. Keep Iterating 


### Data Sources:

* King County Parcel Viewer https://gismaps.kingcounty.gov/parcelviewer2/
* Zillow API https://www.zillow.com/howto/api/APIOverview.htm
    * Received Key
* Walk Score API https://www.walkscore.com/professional/api-sign-up.php
    * Requested API Key
* Seattle Sold Homes
    * Need to find source
* Seattle Median Income by Zip Code https://www.incomebyzipcode.com/washington/

### Target:

* MVP: House sale price matched from 2018 with zillow data only
* Goal: Integrate outside data sources to add further insight to values

### Features:

* Home Attributes  
    * 'zillow_id': '49047113'
    * 'home_type': 'SingleFamily'
    * 'latitude': '47.689225'
    * 'longitude': '-122.378198'
    * 'year_built': '1944'
    * 'property_size': '4532'
    * 'home_size': '1240'
    * 'bathrooms': '1.5'
    * 'bedrooms': '4'
    * 'school_district': 'SEA'

* Assessed Value (zillow or assessor)
    * $ USD
        * 'tax_year': '2017'
        * 'tax_value': '658000.0'
* Sold Value
    * $ USD 
        * 'last_sold_date': '07/18/2014'
        * 'last_sold_price': '520000'
* Walk Score
    * 0-100 indexed score
      * 'walk_score': '87'
      * 'transit_score': '54'
      * 'bike_score': '76'
* Median Income per Zip Code
    * $ USD
      * 'median_income': '102519'

### Things to consider:

* King county assessed value depends on homes around it, but the dependency is unclear.
* Walk Score is a measure of how easy it is to get to bus lines, local shops, etc. I cannot say
for sure what the impact is of 1 bus line or 5 bus lines, or having a grocery store near by is.
