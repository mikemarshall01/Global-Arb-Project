# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:07:24 2018

@author: mima
"""

import pandas as pd
import seaborn as sns
from datetime import datetime as dt
from ImportData import ImportData
from ArbEconomics import arb
from RefineryEconomics import gpw_calculation
import time
        
def crude_destination_check(arb_data):
    """Check that these ports all have corresponding flat rates otherwise they will fail at this stage.
    Also check the crudes are also there. Finally need to check that the load port is in the overall list as known issues with novo etc"""
    ports_with_rates = arb_data.rate_data['LoadPort'].unique()
    crudes_with_codes = arb_data.total.columns.values
    ports_in_targo = arb_data.ports['PortName'].unique()
    
    """Slice our data so only crudes with loadports that have flat rates are returned - this prevents an error"""
    """ This gives us the assyta from the 1st positional argument in var. Then transpose to get crudes as the index and bring back code and loadport"""
    crude_check = pd.DataFrame(arb_data.assay).transpose()[['Code','LoadPort']].dropna(axis=0)
    test_crudes = crude_check.loc[crude_check['LoadPort'].isin(ports_with_rates)]
    no_flat_rates = crude_check.loc[~crude_check['LoadPort'].isin(ports_with_rates)]
    
    """Check to see if the loadports are in the targo extract"""
    test_crudes = test_crudes.loc[test_crudes['LoadPort'].isin(ports_in_targo)]
    not_in_targo = crude_check.loc[~crude_check['LoadPort'].isin(ports_in_targo)]
    
    """Check to see if the neccessary codes are in the total dataframe"""
    try:
        test_crudes = test_crudes.loc[(test_crudes['Code'].isin(crudes_with_codes)) | (test_crudes['Code'] == 'multiple')].drop('Null')
    except:
        pass
    crudes = ['AZERI']
    crudes = list(test_crudes.index.values)
    destinations = ['Rotterdam','Augusta','Houston','Singapore']
    
    return crudes, destinations

   
def make_arbs(crudes,destinations, arb_data): 
    substitution_pct = 0.2 
    counter = 0
    try:
        base_blends = {'Rotterdam':{'URALS NTH':0.5,'EKOFISK':0.2,'FORTIES':0.1,'GULLFAKS':0.1,'STATFJORD':0.1},
            'Augusta':{'QUA IBOE':0.1,'AZERI':0.3,'SAHARAN':0.2,'URALS MED':0.4},
            'Singapore':{'ARAB Medium':0.25,'ARAB LIGHT':0.10,'BASRAH LIGHT':0.15,'CABINDA':0.15,'midland wti':0.1,'AZERI':0.25},
            'Houston':{'MAYA':0.15,'BASRAH LIGHT':0.175,'ARAB Medium':0.175,'eagle ford':0.125,'Bakken':0.125,'midland wti':0.25}}
        
        """generate the landed base blends for each region"""
        base_costs = pd.DataFrame()
        for j in base_blends.keys():
            base_landed = pd.DataFrame()
            for i in base_blends[j].keys():
                #i = 'Basrah Light'
                #j = 'Houston'
                arb_frame = arb(i,j,arb_data)
                freight_list = [column_header for column_header in arb_frame.columns if 'landed' in column_header]
                base_landed[i] = arb_frame[freight_list].mean(axis=1) * base_blends[j][i]
            base_costs[j] = base_landed.sum(1)           
    except Exception as e: print(e)
    
    for m in arb_data.refinery_configs.keys():
        refinery_config = arb_data.refinery_configs[m]
        for i in crudes:
            for k in destinations:
                if 'Iran' in i and 'Houston' in k:
                    pass
                elif 'Olmeca' in i:
                    pass                
                else:
                    try:
                        if counter == 0:
                            m = 'simple'
                            i = 'AZERI'
                            k = 'Singapore'
                            refinery_config = arb_data.refinery_configs['simple']
                            arb_values = arb(i,k, arb_data)
                            base = k[:4]+'_base_landed'
                            arb_values[base] = base_costs[k]
                            gpw = gpw_calculation(i,k, arb_data, refinery_config)
                            arb_values = pd.concat([arb_values,gpw], axis=1)
                            margin_headers = []
                            for j in [index for index in arb_values.columns if '_landed' in index]:
                                name = j[:4]+"_margin"
                                margin_headers.append(name)
                                if 'base' in j:
                                    arb_values[name] =  arb_values['Base_GPW'] - arb_values[j] - arb_values['outright']
                                else:
                                    arb_values[name] =  arb_values['GPW'] - arb_values[j] - arb_values['outright']
                            #arb_values.columns
                            for x in range(len(margin_headers)-1):
                                name = margin_headers[x]+'_advantage'
                                arb_values[name] = (arb_values[margin_headers[x]] - arb_values[margin_headers[-1]]) * substitution_pct
                            arb_values.columns = pd.MultiIndex.from_product([[m],[i],[k], arb_values.columns])
                            counter +=1
                        else:
                            next_arb = arb(i,k, arb_data)
                            base = k[:4]+'_base_landed'
                            next_arb[base] = base_costs[k]
                            next_gpw = gpw_calculation(i,k, arb_data, refinery_config)
                            next_arb = pd.concat([next_arb,next_gpw], axis=1)
                            margin_headers = []
                            for j in [index for index in next_arb.columns if '_landed' in index]:
                               name = j[:4]+"_margin"
                               margin_headers.append(name)
                               if 'base' in j:
                                   next_arb[name] =  next_arb['Base_GPW'] - next_arb[j] - next_arb['outright']
                               else:
                                   next_arb[name] =  next_arb['GPW'] - next_arb[j] - next_arb['outright']
                            for x in range(len(margin_headers)-1):
                               name = margin_headers[x]+'_advantage'
                               next_arb[name] = (next_arb[margin_headers[x]] - next_arb[margin_headers[-1]]) * substitution_pct
                            next_arb.columns = pd.MultiIndex.from_product([[m],[i],[k], next_arb.columns])
                            arb_values = pd.concat([arb_values,next_arb], axis=1)
        
                    except Exception as e:
                        if 'Iran' in i and 'Houston' in k:
                            pass
                        else:
                            print(e)
                            print("{} failed into {}".format(i,k))         
    
    return arb_values

def create_arb_matrix():
    arb_data = ImportData()
    crudes, destinations = crude_destination_check(arb_data)
    global_arbs = make_arbs(crudes, destinations, arb_data)
    
    """Assign the levels names"""
    global_arbs.columns.names, global_arbs.index.names = ['RefineryConfig','Grade','Region','Series'], ['Date']
    global_arbs = global_arbs.round(2)
    return global_arbs
    
    

if __name__ == "__main__":
    #crude = 'AZERI'
    #destination = 'Houston'
    t5 = time.process_time()
    test = create_arb_matrix()
    print("ArbData created successfully: Time was {}".format(time.process_time() - t5))   
    #writer = pd.ExcelWriter('L://TRADING//ANALYSIS//Python//GlobalArbs_FINAL_CHECK12.xlsx')   
    #test.to_excel(writer, sheet_name='global_arbs')
    #writer.save()
    
'''
def upload
tester1 = global_arbs.unstack().reset_index().rename(columns={0:'Value'})
tester1.columns.values
'''


"""
basrah_suez = global_arbs.iloc[global_arbs.index > dt(2017,9,1),(global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['Basrah Light','Basrah Heavy']))&
                 (global_arbs.columns.get_level_values('Region').isin(['Rotterdam','Augusta']))]


import pandas as pd
import numpy as np


columns = [["A","B","C","D","E"],
           ["f8gh", "ht6k", "gf4h", "fs2e", "po9a"]
          ]

c_tuples = list(zip(*columns))
multi_columns = pd.MultiIndex.from_tuples(c_tuples)

index = [[11869, 14363, 29554, 34554, 52473, 62948], 
         [14414, 29806, 31109, 36081, 54936, 63887], 
         ["+", "-", "+", "-", "+", "+"],
         ["1L1", "7P", "02-11", "38A", "4P", "11P"]]

i_tuples = list(zip(*index))
multi_index = pd.MultiIndex.from_tuples(i_tuples)


data = [10, 9,2,7,7,2,7,8,6,5,1,7,2,9,10,6,2,4,10,7,4,6,4,4,3,2,6,9,7,5]

pd.DataFrame(np.array(data).reshape((len(multi_index),len(multi_columns))), 
             index=multi_index,
             columns=multi_columns)
"""




'''

# =============================================================================
#writer = pd.ExcelWriter('C://Users//mima//Documents//GlobalArbs20.xlsx')
writer = pd.ExcelWriter('L://TRADING//ANALYSIS//Python//Basrah3.xlsx')
basrah_suez.to_excel(writer, sheet_name='BasrahLanded')
# =============================================================================
# suez_margins.to_excel(writer, sheet_name='Suez Adv')
# afra_margins.to_excel(writer, sheet_name='Afra Adv')
# =============================================================================
writer.save()

#global_arbs.iloc[:, global_arbs.columns.get_level_values('Series')=='Suez_margin_advantage']

dtd_landed_comparison = global_arbs.iloc[global_arbs.index > dt(2018,1,1),(global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars','Forties','Ekofisk','Saharan']))&
                 (global_arbs.columns.get_level_values('Region')=='Rotterdam')]







suez_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3)=='Suez_margin_advantage']

for i in crudes:
    for j in refinery_configurations.keys():
        try:
            suez_margins[j][i].iloc[suez_margins.index > dt(2018,1,1)].rolling(5, min_periods=1).mean().plot(title=i+' '+j)
        except Exception as e: print('{} passed'.format(i))
        




dtd_landed_comparison = global_arbs.iloc[global_arbs.index > dt(2018,1,1),(global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars','Forties','Ekofisk','Saharan']))&
                 (global_arbs.columns.get_level_values('Region')=='Rotterdam')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_landed_vs_dtd')].groupby(level = ['Grade'], axis=1).mean()

dtd_landed_comparison.plot(title='Rotterdam Landed')

wti_landed_comparison = global_arbs.iloc[global_arbs.index > dt(2018,1,1),(global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars','Forties','Ekofisk','Saharan']))&
                 (global_arbs.columns.get_level_values('Region')=='Houston')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_landed_vs_wti')].groupby(level = ['Grade'], axis=1).mean()

wti_landed_comparison.plot(title='Houston Landed')





wti_landed_comparison = global_arbs.iloc[global_arbs.index > dt(2018,1,1),(global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars','Forties','Ekofisk','Saharan']))&
                 (global_arbs.columns.get_level_values('Region')=='Rotterdam')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_landed_vs_dtd')].groupby(level = ['Grade'], axis=1).mean()













suez_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3).isin(['Suez_landed_vs_wti','Suez_landed_vs_dtd'])]


"""This creates the margins based on a base blend for each region"""
base_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3).isin([
        'Hous_margin','Augu_margin','Rott_margin'])].rolling(5, min_periods=1).mean().groupby(level=['RefineryConfig','Series'], axis=1).mean()  
base_margins.plot(figsize = (7.5,7.5))




suez_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3)=='Hous_margin']

suez_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3)=='Suez_margin_advantage']


        
        
global_arbs['simple']['WTI Midland']['Rotterdam'].loc[suez_margins.index > dt(2018,1,1),:]       
        
        
        
        
global_arbs.iloc[global_arbs.index > dt(2018,1,1) , (global_arbs.columns.get_level_values('RefineryConfig').isin(['complex','simple']))&
                                  (global_arbs.columns.get_level_values('Series')=='Suez_margin_advantage')].rolling(5, min_periods=1).mean().plot(figsize = (7.5,7.5))        
        
        
    

global_arbs.iloc[global_arbs.index > dt(2018,1,1) , (global_arbs.columns.get_level_values('RefineryConfig').isin(['complex','simple']))&
                 (global_arbs.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars']))&
                 (global_arbs.columns.get_level_values('Region')=='Houston')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_margin_advantage')].rolling(5, min_periods=1).mean().plot(figsize = (7.5,7.5))




analysis = global_arbs.iloc[global_arbs.index > dt(2018,1,1) , (global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['LLS']))&
                 (global_arbs.columns.get_level_values('Region')=='Houston')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_landed_vs_dtd')].rolling(5, min_periods=1).mean()


analysis = global_arbs.iloc[global_arbs.index > dt(2018,1,1) , (global_arbs.columns.get_level_values('RefineryConfig')=='simple')&
                 (global_arbs.columns.get_level_values('Grade').isin(['LLS']))&
                 (global_arbs.columns.get_level_values('Region')=='Houston')&
                 (global_arbs.columns.get_level_values('Series')=='Suez_landed_vs_dtd')].rolling(5, min_periods=1).mean()



global_arbs.get_level_values(3)

analysis_picture = analysis.plot(figsize = (7.5,7.5))
analysis.pct_change(periods=10).iloc[-100:,:].plot()
analysis.iloc[-100:,:].plot()

analysis.pct_change(periods=10).loc[(analysis["Suez_margin_advantage"].shift() < 0) & (analysis["Suez_margin_advantage"] > 0)] *= -1

(analysis.iloc[-3,:] - analysis.iloc[-5,:]) / analysis.iloc[-5,:]






random.plot()

random.rolling(5, min_periods=1).mean()

"""Levels are 0: ref type, 1: crude, 2: destination, 3: price series"""


afra_margins = global_arbs.iloc[:, global_arbs.columns.get_level_values(3)=='Afra_margin_advantage']

# =============================================================================
#writer = pd.ExcelWriter('C://Users//mima//Documents//GlobalArbs20.xlsx')
writer = pd.ExcelWriter('L://TRADING//ANALYSIS//Python//SaharanRott1.xlsx')
dtd_landed_comparison.to_excel(writer, sheet_name='Arbs')
# =============================================================================
# suez_margins.to_excel(writer, sheet_name='Suez Adv')
# afra_margins.to_excel(writer, sheet_name='Afra Adv')
# =============================================================================
writer.save()


#suez_margins['simple']['Azeri'].iloc[suez_margins.index > dt(2016,1,1)].plot()

#global_arbs['simple'][['BaseBlendUSGC','CPC Blend']]['Houston'][['diff','Suezmax']].iloc[suez_margins.index > dt(2016,1,1)].plot()

        
        
# =============================================================================
# 
# 
# global_arbs.head()
# var[3]['AAIRB00']
# 
# global_arbs['simple']['Gullfaks']['Houston'].head()
# 
# 
# suez_weekly = suez_margins.resample('W').mean()
# suez_margins
# 
# =============================================================================
global_arbs.columns.names, global_arbs.index.names = ['RefineryConfig','Grade','Region','Prices'], ['Date']

global_arbs.pivot_table(columns = global_arbs.index, index = )
'''