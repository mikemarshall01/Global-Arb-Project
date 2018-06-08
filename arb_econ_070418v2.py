# -*- coding: utf-8 -*-
"""
Created on Sat Mar 3 18:09:50 2018

@author: mima

Need flat rate table
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
import random

data = pd.ExcelFile('C://Users//mima//Documents//toydata1004.xlsx')
assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
ws = pd.read_excel(data, 'ws')
ws_table = pd.read_excel(data, 'ws_table', header = 1)
rate_data = pd.read_excel(data, 'flat_rate')
prices = pd.read_excel(data, 'prices', header = 1)
paper_prices = pd.read_excel(data, 'paper prices', header = 1)
expiry_table = pd.read_excel(data, 'expiry')
ports = pd.read_excel(data, 'ports')
products = pd.read_excel(data, 'rott products')
sub_to_ws = pd.read_excel(data, 'sub_to_ws', header = None)
sub_to_ws = sub_to_ws.set_index([0]).to_dict()

"""
Take in the crude prices and codes and convert to a dataframe.
We need to take the first 2 rows of the prices with no headers as this will give us the cude name and the code ascociated
Then transpose from rows to columns and rename the columns. This will be for later when we determine crude prices basis desired comaprison
"""
prices_reference = (pd.read_excel(data, 'paper prices', header = None).iloc[0:2,1:]).transpose().rename(columns={0:'Name', 1: 'Code'})


"""
Merge the WS table with the prices table, slice df so 2016 onwards (Flat rates last date is 2015). 
We don't drop rows now as dropping would be dependent on any nans in any column
"""
total = prices.merge(ws_table, how = 'inner', left_index = True, right_index = True)
total = total.merge(paper_prices, how = 'inner', left_index = True, right_index = True)
total = total.iloc[total.index > dt(2015,12,31)]


"""
Clean the column hedaers so no white spcaes - use simple list comprehension and set headers equal to cleaned
"""
cleaned_column_headers = [i.strip() for i in total.columns.values]
total.columns = cleaned_column_headers

"""
Initialise the crude and discharge location - this will be driven by user input or looped over a list. 
Also initialise the temp df with index of total.
Temp df is tol hold the dataseries needed to calculate the freight
"""
crude = 'Azeri'
discharge = 'Houston'
df = pd.DataFrame(index=total.index)

def calculate_flat_rate():
    
    flat_rate_table = pd.DataFrame(rate_data[(rate_data['LoadPort'] == assay[crude]['LoadPort'])&
                     (rate_data['DischargePort'] == discharge)])
    
    return flat_rate_table

flat_rate_table = calculate_flat_rate()





"""
THIS IS ESSENTIALLY A VLOOKUP

inputs are the table to look up values from for each row of another table

in this case we use it to look up the year in the dataframes row and take the year and match it against the corresponding row (or rows? possibly but need to investigate)
and return the value in the rate row.

use the .iat function to return the inger values - this speeds up operations

x refers to the row in the dataframe we are applying to and the name is the index reference of the row - which is the dateindex.
Hence, since you cant take the index.year directly, must turn the date into a string, get the date as a dt object from the string and then get the year

"""

def calculate_flat_rates(flat_rate_table, x):
    flat_rate_row = flat_rate_table[(flat_rate_table["Year"] == dt.strptime(str(x.name), '%Y-%m-%d %H:%M:%S').year)]
    return int(flat_rate_row["Rate"].iat[0])

df['Rate'] = total.apply(lambda x: calculate_flat_rates(flat_rate_table, x), axis =1)



"""
Then we need to calculate the worldscale - for this we need to get the relevent from and to ws rate, regardless of size so can compare suez vs afra etc
We get the first rate by looking up where the loadport attached to our crude matches the load port in the ports list and returns the subregion
As the subregions dont completely match off (i.e. in targo its MED OECD / MED MID EAST etc etc which is all med, we need to map the sub regions to make sure aligned
We then convert the result to a string and we set index = false as we dont want the header row returned
Do the same for the destination region and then slice the ws table which ahs the routes, origins, destinations etc by where the to and from regions match
This gives us the neccessary world scale codes 

We then loop through ecah item in the list annd determine what to do with it. The ws table has a column called terms which states if lumpsum or $/mt
If lump, then we need to ahndle differently - i.e. we take the value of the lumpsum, multiply by 1mn and divide by the size of the vessel against the lumpsum
If normal then we take the WS, divide by 100, multiply by the rate we placed in the temp frame earlier and divide y the crude specific bt factor

Final values will be in $/bbl

"""
  
def calculate_world_scale():
    """This finds the correct worldscale rate and adjusts if it is lumpsum"""
    
    sub_region = ports[ports['Name'] == assay[crude]['LoadPort']]['Subregion']
    sub_region = sub_region.map(sub_to_ws[1]).to_string(index = False)
    
    sub_region_2 = ports[ports['Name'] == discharge]['Subregion']
    sub_region_2 = sub_region_2.map(sub_to_ws[1]).to_string(index = False)
  
    ws_codes = ws[(ws['Origin'] == sub_region) &
                      (ws['Destination'] == sub_region_2)]

    for i in list(ws_codes['Code']):
        name = ws_codes[ws_codes['Code'] == i]['Name'].iat[0]
        if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
            df[name] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['Size'].values * 1000)
        else:
            df[name] = total[i] / 100 * df['Rate'] / assay[crude]['Conversion']
         
    return df

calculate_world_scale()

def pricing_adjustment():
    index_wti = ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI']
    index_dtd = ['DATED	BRENT DATED','N.SEA DATED','BTC Dated',	'MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','	GHANA DATED']
    index_dub = ['DUBAI','DUBAI M2']
    #TEST PRICES
    def generate_price(row, minp, maxp):
        return row + random.uniform(minp,maxp)

    # will need something to determine which cfds to choose - for now thw ones to choose are going to be the 3/4
    
    """
    This function allows us to apply the expiration date for the wti futures used to determine what structure we apply to the CMA
    """
    def apply_expiry(expiry_table, x):
        expiry_row = expiry_table[(expiry_table["Month"].dt.month == dt.strptime(str(x.name), '%Y-%m-%d %H:%M:%S').month)&
                                  (expiry_table["Month"].dt.year == dt.strptime(str(x.name), '%Y-%m-%d %H:%M:%S').year)]
        return expiry_row["Expiry"].iat[0]
    
    df['Expiry'] = df.apply(lambda x: apply_expiry(expiry_table, x), axis =1)

    dtd = total['PCAAS00']
    wtim1 = total['CLc1'] 
    wtim2 = total['CLc2']
    cfd1 = total['PCAKG00']
    cfd2 = total['AAGLV00']
    efpm2 = total['AAGVX00']
    brentm2 = total['LCOc2']
    efs2 = total['AAEBS00']

    def convert_wti():
        if assay[crude]['Index'] in index_wti:
            if (df.index < df.Expiry).any(): 
                df['diff_vs_wti'] = total[assay[crude]['Code']] +  (wtim1 -  wtim2) 
            else:
                df['diff_vs_wti'] = total[assay[crude]['Code']]
                
        elif assay[crude]['Index'] in index_dtd:
            if (cfd1 > cfd2).any(): 
                df['diff_vs_wti'] = total[assay[crude]['Code']] + cfd2 +  efpm2 + (brentm2 - wtim2)
            else: 
                df['diff_vs_wti'] = total[assay[crude]['Code']] + cfd1 +  efpm2 + (brentm2 - wtim2)
                
        elif assay[crude]['Index'] in index_dub:
            pass
        
        else:
           df['diff_vs_wti'] = total[assay[crude]['Code']]
            
        return df['diff_vs_wti']
    
    def convert_dtd():
        
        if assay[crude]['Index'] in index_wti:
            if (df.index < df.Expiry).any():
                if (cfd1 > cfd2).any():
                    df['diff_vs_dtd'] = total[assay[crude]['Code']] +  (wtim1 -  wtim2) - brentm2 - efpm2 - cfd2
                else:
                    df['diff_vs_dtd'] = total[assay[crude]['Code']] +  (wtim1 -  wtim2) - brentm2 - efpm2 - cfd1
            else:
                if (cfd1 > cfd2).any():
                    df['diff_vs_dtd'] = total[assay[crude]['Code']] - brentm2 - efpm2 - cfd2
                else:
                    df['diff_vs_dtd'] = total[assay[crude]['Code']] - brentm2 - efpm2 - cfd1
                    
        elif assay[crude]['Index'] in index_dtd:
            
            if (total.index < dt(2018,7,28)).any():
                if (cfd1 > cfd2).any():              
                    df['diff_vs_dtd'] =  total[assay[crude]['Code']] + cfd2
                else:
                    df['diff_vs_dtd'] =  total[assay[crude]['Code']] + cfd1
            else:
                df['diff_vs_dtd'] =  trader_assessed_prices[assay[crude]['Code']]
                
        elif assay[crude]['Index'] in index_dub:
            pass
        
        else:
            df['diff_vs_dtd'] = total[assay[crude]['Code']]
            
        return df['diff_vs_dtd']
    
    def convert_dub():
        if assay[crude]['Index'] in index_wti:
            if (df.index < df.Expiry).any():
                df['diff_vs_dub']= total[assay[crude]['Code']] +  (wtim1 -  wtim2)  - brentm2 + efs2
            else:
                df['diff_vs_dub'] = total[assay[crude]['Code']] - brentm2 + efs2
                
        elif assay[crude]['Index'] in index_dtd:
            if (cfd1 > cfd2).any():  
                df['diff_vs_dub'] = total[assay[crude]['Code']] + cfd2 +  efpm2 + efs2
            else:
                df['diff_vs_dub'] = total[assay[crude]['Code']] + cfd1 +  efpm2 + efs2
        
        elif assay[crude]['Index'] in index_dub:
            pass
        
        return df['diff_vs_dub'] 
    
    convert_dtd() 
   
#def other_costs("This will have the port costs table and any rebates etc that are needed")