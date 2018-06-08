# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 20:49:41 2018
@author: mike_
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 11:12:26 2018
@author: mima


TO DO: g
1) get/create neccessary tables for flat rate, worldscale, prices
2) give crudes 



"""
from datetime import datetime
import pandas as pd

"""
initialize the tables we will need for the model inputs

flat_rate for flat rate records - sona currently working on - need to figure how to parse a sheet and turn into record sets.
Also, figure this would be better as a function that returns a dataframe of equal length to that of the series - poass the prices.index and attach the dates

ws_table will take in info from database on worldscale routes  - have this in excel format, can mimic database

prices will be a data frame with all the time series prices - just need to make a sheet with the neccessary crude and ws codes to mimic database

need crude load terminal to port to country to sub region to region

need to check the lumpsums 

"""

flat_rate = pd.DataFrame([(1,'BP - Sullom Voe Terminal','Houston',datetime(2018,1,1), 1.5)], columns=['Id','Load Port','Discharge Port','Year','Rate'])

ws_table = pd.DataFrame([(1,'NWE','USGC',1000,'$/T','ABCDEFG'),
                         (1,'NWE','USGC',1000,'$/T','poiuytr'),
                         (1,'NWE','USGC',2000,'lumpsum','1234567')
                         ], columns=['Id','Origin','Destination','Size','Terms','Code'])

prices = pd.DataFrame([(2, 15, 10, 3.3)
                         ], columns=['mike','ABCDEFG','poiuytr','1234567'])





# Temp Prices
crude = 'LLS'
assay[crude]['Index'] = 'CMA'
assay[crude]['Code'] = 'FGHJAK'
assay[crude]['Basis'] = 
assay[crude]['Default_Load_Port']
assay[crude]['Secondary_Load_Port']



wtim1 =
wtim2 = 
brentm1 = 
brentm2 =
dubaim1 = 
dubaim2 =
efpm2 = 
firstcfd = 
secondcfd = 
dated_brent =
efs2 = 


crude_price = 2.5
structure = 0.5
brent_wti_2m_spread = 3.50
brent_dubai_2m = 3.50
brent_efp_M2 = 0.02
cfd = -1.00
date = datetime(2018,2,1)
expiry_date = datetime(2018,3,1)
basis = 'CMA'
port_costs = 0.05
other_costs = 0.5
'NEed to name/label world scale routes and attach to neccessray journies'
# =============================================================================
# ### LANDED COSTS ### tules to get back to benhcmarks
# 
# ### Need 
#         1) expiry dates table
# 
# 
# import dataframe with time series values
# 
# need expiry month table
# =============================================================================
'this gives vs 2nd month WTI, need an expiry dates table'
'Basis US Crudes' # this is to take into account the mismatch between expiry date of the WTI future and the the pricing month of the CMA
# whilst the date is before the expiry, the crude is assessed at diff + WTI M1 hence you add stucture, when that goes off the board 
# the cma month remains the same but as future has expired, it moves forward one month meaning we now have diff vs 2nd month WTI until the calender month ends
"""
*** Have to come up with a rules table for pricing basis ***
i.e. if vs wti futs then have to adjust, if vs latam dated strip then adjust cfds etc
"""


#""" This is because early in month, its against wti cm+1, on expiry, the futs become cm+2,
#            so no need to add structure"""
#            if prices.index < prices.wti_expiry_date:
#               """crude price + wti structure"""
#    if port_mappings[assay[crude]['Default_Load_Port']]

def pricing_adjustment(crude):
    index_wti = ['CMA','WTI Forward','WTI Front Line']
    index_dtd = ['CMA','WTI Forward','WTI Front Line']
    index_dubai = ['CMA','WTI Forward','WTI Front Line']
    
    def convert_wti(crude):
        if assay[crude]['Index'] in index_wti:
            if prices.index < prices.wti_expiry_date: 
                diff_vs_wti = prices[assay[crude]['Code']] +  (prices[wtim1] -  prices[wtim2]) 
            else:
                diff_vs_wti = prices[assay[crude]['Code']]
                
        elif assay[crude]['Index'] in index_dtd:
            if prices[first_cfd] > prices[second_cfd]: 
                diff_vs_wti = prices[assay[crude]['Code']] + prices[second_cfd] +  prices[efpm2] + (prices[brentm2] - prices[wtim2])
            else: 
                diff_vs_wti = prices[assay[crude]['Code']] + prices[first_cfd] +  prices[efpm2] + (prices[brentm2] - prices[wtim2])
                
        elif assay[crude]['Index'] in index_dub:
            pass
        
        else:
            diff_vs_wti = prices[assay[crude]['Code']] - prices[wtim2]
            
        return diff_vs_wti
    
    def convert_dtd(crude):
        if assay[crude]['Index'] in index_wti:
            if prices.index < prices.wti_expiry_date:
                if prices[first_cfd] > prices[second_cfd]:
                    diff_vs_dtd = prices[assay[crude]['Code']] +  (prices[wtim1] -  prices[wtim2]) - prices[brentm2] - prices[efpm2] - prices[second_cfd]
                else:
                    diff_vs_dtd = prices[assay[crude]['Code']] +  (prices[wtim1] -  prices[wtim2]) - prices[brentm2] - prices[efpm2] - prices[first_cfd]
            else:
                if prices[first_cfd] > prices[second_cfd]:
                    diff_vs_dtd = prices[assay[crude]['Code']] - prices[brentm2] - prices[efpm2] - prices[second_cfd]
                else:
                    diff_vs_dtd = prices[assay[crude]['Code']] - prices[brentm2] - prices[efpm2] - prices[first_cfd]
                    
                    
        """ convert the dtd related crude into an equivalent landing period vs other crudes - could do prompt but have talked to Andrea about this"""
        elif assay[crude]['Index'] in index_dtd: # This will have differing cfd values dependign on location / journey time
            if prices.index < datetime(2018,7,28):
                if prices[first_cfd] > prices[second_cfd]:              
                    diff_vs_dtd =  prices[assay[crude]['Code']] + prices[second_cfd]
                else:
                    diff_vs_dtd =  prices[assay[crude]['Code']] + prices[first_cfd]
            else:
                diff_vs_dtd =  trader_assessed_prices[assay[crude]['Code']]
                
        elif assay[crude]['Index'] in index_dub:
            pass
        
        else:
            diff_vs_dtd = prices[assay[crude]['Code']] - prices[dated_brent]
            
        return diff_vs_dtd
    
    def convert_dub(crude):
        if assay[crude]['Index'] in index_wti:
            if prices.index < prices.wti_expiry_date:
                diff_vs_dubai = prices[assay[crude]['Code']] +  (prices[wtim1] -  prices[wtim2]) - prices[brentm2] + prices[efs2]
            else:
                diff_vs_dubai = prices[assay[crude]['Code']] - prices[brentm2] + prices[efs2]
                
        elif assay[crude]['Index'] in index_dtd:
            if prices[first_cfd] > prices[second_cfd]:  
                diff_vs_dubai = prices[assay[crude]['Code']] + prices[second_cfd] +  prices[efpm2] + prices[efs2]
            else:
                diff_vs_dubai = prices[assay[crude]['Code']] + prices[first_cfd] +  prices[efpm2] + prices[efs2]
        
        elif assay[crude]['Index'] in index_dub:
            pass
        
        return diff_vs_dub
        
        
            
            
            
"""
WAF - > Eur is (wk 3 - wk 5) / 14 * journey time
US - > Eur 
"""
           
 
port_costs = port_costs + other_costs
""" 
Create a table with the orldscale routes and codes ina tuple and create a line of code that can look at the tuple, pick the correct to and from locations
The origin will be determined by the crude loading location which will part of the crude assay, this can be overridden if needs be.
The load port along with the discharge REGION will then be used to get the correct flat rates i.e. brent to usgc will have sullum voe to houston - houston as the default
discharge port for that region. This will then pull the correct flat rate (based on the year) from out flatrate table. The worldscale will be picked based on the SUB-REGIONS
we assign to the wordscale table and 
"""
"""
Region Tree = crude / products
IdLocation links to location table and is the load port
Id Map Type is what typre of mapping it is, i.e. refinery / grade etc
IdMap is tyhe ID whatever I am mappign - depends what map type is, IdMap will look in a different table 
"""

"""
Here is the function for the calculating the needed flat rate
"""

crude = 'Agbami'
assay = {'Agbami':crude_assay}
assay['Agbami']['LoadPort'] = 'BP - Sullom Voe Terminal'

dischargeport = 'Houston'
rate = flat_rate[(flat_rate['Load Port'] == assay[crude]['LoadPort'])&
                     (flat_rate['Discharge Port'] == dischargeport)&
                     (price.index.year == flat_rate['Date'])]['Rate'].values
                 
"""We have the crude price vs the destination index and the flat rate and port costs, just need WS
assume we have a tvle with loaport, load sub region, discharge port and discharge sub region
I want to get the corresponding world scale code  - i need the load sub region, dis sub - figure out size later,
would need to see all anyway"""

# get the worldscale codes from the table

"""MESSSY CODE - but have managed to get the look up into the worldscale table that returnd=s the neccessary worldscale codes,
then move onto attempting to account for the lump sum difference """


assay[crude]['SubRegion'] = 'NWE'
codes = ws_table[(ws_table['Origin'] == assay[crude]['SubRegion']) &
                  (ws_table['Destination'] == 'USGC')]
#codes[codes['Code'] == 'ABCDEFG']
[]

assay[crude]['Code'] = 'mike'
assay[crude]['BT'] = 6.9

freight_frame = pd.DataFrame()
#This gives me the 
for i in list(codes['Code']):
    if codes[codes['Code'] == i]['Terms'].values == 'lumpsum':
        dollars_barrel = prices[codes[codes['Code'] == i]['Code'].values] * 1000000 / (codes[codes['Code'] == i]['Size'].values * 1000)
        x = diff_vs_dated + dollars_barrel + port_costs
        landed_costs.append(x)
    else:
        x = diff_vs_dated + port_costs + prices[i] / 100 * flat_rate / assay[crude]['BT']
        landed_costs.append(x)












""" this gtes you the columns from the dataframe"""
prices[codes]
                
                

def flat_rate(crude, dischargeport):
    """Find loadport of crude passed"""
    loaport = flat_rate
    
   

ws = pd.read_excel('L://TRADING//ANALYSIS//GLOBAL//Arb Models//GPW model RA.xlsm', sheetname = 'WS', header = [0], index_col = [0])
# gets the column headers and the codes as one header
arrays = [ws.columns.values,ws.iloc[0,:].values]
tuples = list(zip(*arrays))
index = pd.MultiIndex.from_tuples(tuples, names=['name', 'code'])
prices = pd.read_excel('L://TRADING//ANALYSIS//GLOBAL//Arb Models//GPW model RA.xlsm', sheetname = 'Price', header = [0,1])
ws.iloc[0,:]
ws.columns = pd.MultiIndex.from_product([ws.columns.values, ws.iloc[0,:].values])
'Will need to link in worldscale routes - go through and assign based on load / discharge region'
flat_rate = 17.59 # currently the Houston to Rotterdam number - this will eventually link from the sql table where we will keep port to port rates
freight_to_port = 0 
bt_factor = assay[crude]['Conversion']
afra_route_ws
suez_route_ws
v_route_ws
cargo_size
lumpsum

afra_freight = port_costs + flat_rate / bt_factor * afra_route_ws / 100
suez_freight = port_costs + flat_rate / bt_factor * suez_route_ws / 100
v_freight = port_costs + flat_rate / bt_factor * v_route_ws / 100
lump_freight = port_costs + lumpsum / cargo_size / bt_factor
'Think could use a dictionary? i.e. {Houston: {Afra : Afra}
'                                               {Suez: Suez} }'

if WS route:
    if aframax:
        freight = freight_to_port + flat_rate / bt_factor * route_ws / 100 # suez or afra needs to be seperated
    elif: suezmax:
        ...
else:
    # LumpSum freight - lumpsum in millions and cargo size in tonnes
    freight = freight_to_port + lumpsum / cargo_size / bt_factor
# for Korea rebate where they take the neccessary rebate vs freight from Murban to Yosu
    freight = np.maximum(np.minimum(freight - murban_freight,0.60),0)

current_landed = current_landed + freight + port_costs
     
       








