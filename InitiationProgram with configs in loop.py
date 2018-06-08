# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:07:24 2018

@author: mima
"""

import pandas as pd
from ArbEcons2504 import import_data
from ArbEcons2504 import arb
from GPW2504 import gpw_calculation

var = import_data()

""" This gives us the assyta from the 1st positional argument in var. Then transpose to get crudes as the index and bring back code and loadport"""
crude_check = pd.DataFrame(var[0]).transpose()[['Code','LoadPort']].dropna(axis=0)
#len(test_crudes)
"""Check that these ports all have corresponding flat rates otherwise they will fail at this stage.
Also check the crudes are also there. Finally need to check that the load port is in the overall list as known issues with novo etc"""
ports_with_rates = var[4]['LoadPort'].unique()
crudes_with_codes = var[3].columns.values
ports_in_targo = var[2]['Name'].unique()


"""Slice our data so only crudes with loadports that have flat rates are returned - this prevents an error"""
test_crudes = crude_check.loc[crude_check['LoadPort'].isin(ports_with_rates)]
no_flat_rates = crude_check.loc[~crude_check['LoadPort'].isin(ports_with_rates)]

"""Check to see if the loadports are in the targo extract"""
test_crudes = test_crudes.loc[test_crudes['LoadPort'].isin(ports_in_targo)]
not_in_targo = crude_check.loc[~crude_check['LoadPort'].isin(ports_in_targo)]

"""Check to see if the neccessary codes are in the total dataframe"""
test_crudes = test_crudes.loc[test_crudes['Code'].isin(crudes_with_codes)]
codes_missing = crude_check.loc[~crude_check['Code'].isin(crudes_with_codes)]


#crudes = list(test_crudes.index.values)
crudes = ['Brent','Azeri','CPC Blend']
#destinations = ['Houston']

#destinations = ['Houston','Rotterdam','Augusta','Singapore']

        
destinations = ['Rotterdam','Augusta','Houston']
#crudes = ['Brent','Azeri','CPC Blend']

refinery_configurations = {'simple':{'refinery_volume':200,
                             'reformer_capacity':42,
                             'fcc_capacity':48,
                             'coker_capacity':0,
                             'lvn_gasolinepool':0.12,
                             'kero_gasolinepool':0.15},
                    'complex':{'refinery_volume':200,
                             'reformer_capacity':42,
                             'fcc_capacity':48,
                             'coker_capacity':20,
                             'lvn_gasolinepool':0.12,
                             'kero_gasolinepool':0.15}} 


#refinery_config = refinery_configurations['simple']

def make_arbs(crudes,destinations, *var, **refinery_configurations): 
    counter = 0
    
    base_blends = {'Rotterdam':{'Urals Nth':0.5,'Ekofisk':0.2,'Forties':0.1,'Gullfaks':0.1,'Statfjord':0.1},
        'Augusta':{'Qua Iboe':0.1,'Azeri':0.3,'Saharan':0.2,'Urals Med':0.4}}
        #'Houston':{'Canadian Heavy':0.1,'Castilla':0.15,'Basrah Light':0.125,'Arab Medium':0.125,'LLS':0.25,'Qua Iboe':0.125,'WTI Midland':0.125}}
    
    """generate the landed base blends for each region"""
    base_costs = pd.DataFrame()
    for j in base_blends.keys():    
        for i in base_blends[j].keys():
            base_landed = pd.DataFrame()
            arb_frame = arb(i,j,*var)
            freight_list = [column_header for column_header in arb_frame.columns if 'landed' in column_header]
            base_landed[i] = arb_frame[freight_list].mean(axis=1) * base_blends[j][i]
            base_costs[j] = base_landed.sum(1)
    
    for m in refinery_configurations.keys():
        refinery_config = refinery_configurations[m]
        for i in crudes:
            for k in destinations:
                try:
                    if counter == 0:
                        arb_values = arb(i,k, *var)
                        base = k[:4]+'_base_landed'
                        arb_values[base] = base_costs[k]
                        gpw = gpw_calculation(i,k, *var, **refinery_config)
                        arb_values = pd.concat([arb_values,gpw], axis=1)
                        margin_headers = []
                        for j in [index for index in arb_values.columns if '_landed' in index]:
                            name = j[:4]+"_margin"
                            margin_headers.append(name)
                            if 'base' in j:
                                arb_values[name] =  arb_values['Base_GPW'] - arb_values[j] - arb_values['outright']
                            else:
                                arb_values[name] =  arb_values['GPW'] - arb_values[j] - arb_values['outright']
                        for x in range(len(margin_headers)-1):
                            name = margin_headers[x]+'_advantage'
                            arb_values[name] = arb_values[margin_headers[x]] - arb_values[margin_headers[-1]]
                        arb_values.columns = pd.MultiIndex.from_product([[m],[i],[k], arb_values.columns])
                        counter +=1
                    else:
                        next_arb = arb(i,k, *var)
                        base = k[:4]+'_base_landed'
                        next_arb[base] = base_costs[k]
                        next_gpw = gpw_calculation(i,k, *var, **refinery_config)
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
                           next_arb[name] = next_arb[margin_headers[x]] - next_arb[margin_headers[-1]]
                        next_arb.columns = pd.MultiIndex.from_product([[m],[i],[k], next_arb.columns])
                        arb_values = pd.concat([arb_values,next_arb], axis=1)
    
                except Exception as e:
                    print(e)
                    print("{} failed into {}".format(i,k))         
    
    return arb_values

global_arbs = make_arbs(crudes, destinations, *var, **refinery_configurations)
  
writer = pd.ExcelWriter('C://Users//mima//Documents//GlobalArbs7.xlsx')
global_arbs.to_excel(writer, sheet_name='Arbs')
writer.save()
