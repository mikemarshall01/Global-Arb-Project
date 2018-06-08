# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:46:57 2018

@author: mima
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:49:51 2018

@author: mima
"""
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from collections import defaultdict
from datetime import datetime as dt
from ArbEcons import import_data

# =============================================================================
# # Get the Data
# data = pd.ExcelFile('C://Users//mima//Documents//toydata1004.xlsx')
# assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
# #raw_assay = pd.read_excel('L://TRADING//ANALYSIS//GLOBAL//Arb Models//GPW model RA.xlsm', sheetname = 'Upload_Test', header = 0, index_col = 'Database_Name')
# 
# # with the crude name set as the index, we use pd.to_dict to set a dictioanry of dictionaries
# crude = 'Amna'
# 
# 
# 
# paper_prices = pd.read_excel(data, 'paper prices', header = 1)
# prices = pd.read_excel(data, 'prices', header = 1)
# total = prices.merge(ws_table, how = 'inner', left_index = True, right_index = True)
# total = total.merge(paper_prices, how = 'inner', left_index = True, right_index = True)
# total = total.iloc[total.index > dt(2015,12,31)]
# =============================================================================

assay, ws, ports, total, rate_data, sub_to_ws, df = import_data()

#crude = 'Forties'
#destination = 'HOUSTON'

def cdu(crude, refinery_volume = None, lvn_gasolinepool = None, kero_gasolinepool = None):
    ### INPUT CRUDE ###
    #crude = 'Forties'
    #refinery_volume = 200
    
    ### Fractions for gaosline pool
    if refinery_volume is None:
        refinery_volume = 200
    if lvn_gasolinepool is None:
        lvn_gasolinepool = 0.12
    if kero_gasolinepool is None:
        kero_gasolinepool = 0.15
        
    hvn_input = assay[crude]['HVN'] * refinery_volume
    kero_input = assay[crude]['KERO'] * kero_gasolinepool * refinery_volume
    reformer_input = hvn_input + kero_input
    vgo_input = assay[crude]['VGO'] * refinery_volume
    
    return refinery_volume, reformer_input, kero_input, hvn_input, vgo_input, lvn_gasolinepool, kero_gasolinepool

def reformer(refinery_volume, reformer_input, kero_input, hvn_input, reformer_capacity = None):
    
    #reformer_capacity = 42
    if reformer_capacity is None:
        reformer_capacity = refinery_volume*0.21
    reformer_output = {}
    reformer_assay_standard = {
        'c3': {'yield': 0.025},
        'c4': {'yield': 0.025},
        'h2': {'yield': 0.0},
        'btx': {'yield': 0.0},
        'gasoline': {'yield': 0.95}
        }
    utilised_ref_cap = reformer_capacity * 0.97
    reformer_volume = np.minimum(utilised_ref_cap, reformer_input)
    reformer_output['surplus_for_jet'] = np.maximum(np.minimum(reformer_input - utilised_ref_cap, kero_input), 0)
    reformer_output['surplus_for_naphtha'] = np.maximum(hvn_input - utilised_ref_cap, 0)
    reformer_output['propane'] = reformer_assay_standard['c3']['yield'] * reformer_volume 
    reformer_output['butane'] = reformer_assay_standard['c4']['yield'] * reformer_volume 
    reformer_output['h2'] = reformer_assay_standard['h2']['yield'] * reformer_volume
    reformer_output['btx'] = reformer_assay_standard['btx']['yield'] * reformer_volume
    reformer_output['gasoline'] = reformer_assay_standard['gasoline']['yield'] * reformer_volume
    
    return reformer_output

def fcc(refinery_volume, vgo_input, fcc_capacity = None):
    #fcc_capacity = 48
    if fcc_capacity is None:
        fcc_capacity = refinery_volume*0.21
    fcc_output = {}
    fcc_assay_standard = {
        'c3': {'yield': 0.1},
        'c4': {'yield': 0.01},
        'c3__': {'yield': 0.02},
        'lco': {'yield': 0.20},
        'clo': {'yield': 0.20},
        'gasoline': {'yield': 0.56}
        }

    utilised_fcc_cap = fcc_capacity * 0.97 # this is 97% ulitilisation of total FCC capacity
    fcc_volume = np.minimum(utilised_fcc_cap, vgo_input)
    
    fcc_output['surplus_vgo'] = np.maximum(vgo_input - utilised_fcc_cap, 0)
    fcc_output['propane'] = fcc_assay_standard['c3']['yield'] * fcc_volume
    fcc_output['butane'] = fcc_assay_standard['c4']['yield'] * fcc_volume
    fcc_output['c3__'] = fcc_assay_standard['c3__']['yield'] * fcc_volume
    fcc_output['lco']  = fcc_assay_standard['lco']['yield'] * fcc_volume
    fcc_output['clo']  = fcc_assay_standard['clo']['yield'] * fcc_volume
    fcc_output['gasoline'] = fcc_assay_standard['gasoline']['yield'] * fcc_volume
    
    return fcc_output

def fo_blend(crude, refinery_volume, fcc_output):
    residue = {}
    residue['volume'] = assay[crude]['RESIDUE'] * refinery_volume
    
    # HDS removing suplhur from the bbl before processing
    low_HDS_factor = 0.5
    high_HDS_factor = 0.8
    if assay[crude]['RESIDUE_sulphur'] < 0.035:
        residue['sulphur'] = assay[crude]['RESIDUE_sulphur'] * high_HDS_factor
    else:
        residue['sulphur'] = assay[crude]['RESIDUE_sulphur'] * low_HDS_factor
    
    residue['density'] = assay[crude]['RESIDUE_density'] 
    residue['sulphurW'] = residue['volume'] * residue['sulphur'] * residue['density'] * 1000 / 6.29 # convert the bbls volume to tonnes
    residue['viscosity_index'] = 23.1 + np.log10(np.log10(assay[crude]['RESIDUE_v50']+0.8))*33.47 # this is a set conversion calculation
    
    # certain vis and FO sulphur % avail. We want to use the low sulphur slurry oil or if need the LCO to blend down the resid
    imported_vgo = 0  
    diluent_used = 0       
    diluent = {
            'volume': fcc_output['surplus_vgo'] + fcc_output['clo'] + imported_vgo,
            'sulphur': 0.001,
            'density': 0.897,
            'viscosity': 3,
            'lco_for_blending': 0,
            'target_sulphur': 0,
            'volume_used': diluent_used
            }
    
    diluent['viscosity_index'] = 23.1 + np.log10(np.log10(diluent['viscosity'] + 0.8)) * 33.47
    diluent['sulphurW'] = diluent['volume'] * diluent['sulphur'] * diluent['density'] * 1000 / 6.29
    

        # create the 'cost' function - what do we want to minimise?
    def target_viscosity(diluent_used):
        return 10 ** (10 ** (((diluent_used / (diluent_used + residue['volume']) * diluent['viscosity_index'] + residue['volume'] / (diluent_used + residue['volume']) * residue['viscosity_index'])-23.1)/33.47)) -0.8
    
    # we want to minimise the target viscosity subject to i) the amount of diluent we have avliable to blend and ii) giving away as little as possible so cst at 375
    cons = ({'type': 'ineq', 'fun': lambda diluent_used: diluent['volume'] - diluent_used},
             {'type': 'ineq', 'fun': lambda diluent_used: target_viscosity(diluent_used) - 380})  
 
    # save the output from the optimisation function to res then take the optimized value and pass it into diluent_used
    res = minimize(lambda diluent_used: target_viscosity(diluent_used), [0], method = 'COBYLA', constraints=cons)     
    diluent['volume_used'] = res.x[0]
    diluent['cst'] = res.fun

    # =============================================================================
    # Left blank for the addition of if vgo is needed to be imported
    # Essentially need to take what the vicosity etc is after the first minimise function and then minimised again without the first constraint 
    # 
    # if target_viscosity(diluent_used) > 380:
    #     minimize(lambda diluent_used: target_viscosity(diluent_used), [0], method = 'COBYLA', constraints=cons)
    #     
    # =============================================================================

    diluent['surplus_after_blend'] = diluent['volume'] - diluent['volume_used'] # need to have an optimizer function of some sort                        
    diluent['sulphur_per_bbl'] = diluent['volume_used'] * diluent['density'] * 1000 / 6.29 * diluent['sulphur']
    diluent['weight'] = diluent['volume_used'] * diluent['density'] * 1000 / 6.29
    residue['sulphur_per_bbl'] = residue['volume'] * residue['density'] * 1000 / 6.29 * residue['sulphur']
    residue['weight'] = residue['volume']  * residue['density'] *1000 / 6.29

    combined_fo_sulphur = diluent['sulphur_per_bbl'] + residue['sulphur_per_bbl'] 
    combined_fo_sulphur_weight = diluent['weight'] + residue['weight']
    diluent['target_sulphur'] = combined_fo_sulphur / combined_fo_sulphur_weight
    
    return diluent

def calculate_yields(crude, refinery_volume, reformer_output, fcc_output, lvn_gasolinepool, kero_gasolinepool, diluent):
    
    yields = {}  

    yields['propane'] = (assay[crude]['LPG'] * refinery_volume * 0.5
                           + reformer_output['propane'] 
                           + fcc_output['propane'] 
                           + fcc_output['c3__'] * 0.5
                           ) / refinery_volume
    
    yields['butane'] = (assay[crude]['LPG'] * refinery_volume * 0.5
                       + reformer_output['butane'] 
                       + fcc_output['butane']
                       + fcc_output['c3__'] * 0.5
                       ) / refinery_volume
                
    yields['naphtha'] = (assay[crude]['LVN'] * refinery_volume * (1-lvn_gasolinepool)
                           + reformer_output['surplus_for_naphtha']
                           ) / refinery_volume
                                                
    #yields['btx'] = (reformer_output['btx']) / refinery_volume
    
    yields['gasoline'] = (reformer_output['gasoline'] + fcc_output['gasoline']) / refinery_volume
    yields['kero'] = (assay[crude]['KERO'] * (1-kero_gasolinepool) * refinery_volume + reformer_output['surplus_for_jet']) / refinery_volume
    yields['ulsd'] = ((assay[crude]['LGO'] + assay[crude]['HGO']) * refinery_volume) / refinery_volume
    yields['gasoil'] = (fcc_output['lco'] - diluent['lco_for_blending'] + diluent['surplus_after_blend']) / refinery_volume
    yields['lsfo'] = (diluent['volume_used'] + assay[crude]['RESIDUE'] * refinery_volume) / refinery_volume
    
# =============================================================================
#     yields['f_l'] = 1- (yields['propane']
#                         + yields['butane']
#                         + yields['naphtha']
#                         #+ yields['btx'] 
#                         + yields['gasoline'] 
#                         + yields['kero'] 
#                         + yields['ulsd'] 
#                         + yields['gasoil'] 
#                         + yields['lsfo']) 
# =============================================================================

    return yields


def calculate_margin(yields, destination, ports, sub_to_ws):

    ''' this is to create the look up neccessary for generating the table'''

    price_codes = {'HOUSTON':{'propane':'PCALB00',
                      'butane':'ASGCB00',
                      'naphtha':'AASAV00',
                      'gasoline':'AAGXH00',
                      'kero':'AAWEY00',
                      'ulsd':'AASAT00',
                      'gasoil':'AAUFL00',
                      'lsfo':'AAHPM00'}}
   
    pricing_centre = ports[ports['Name'].str.lower() == destination.lower()]['Subregion'].map(sub_to_ws[3]).to_string(index = False)   
    
    yield_codes = {}
    for i, j in yields.items():
        yield_codes[price_codes[pricing_centre][i]] = j

    regional_price_set = total[list(yield_codes.keys())].fillna(method='bfill')
    gpw = regional_price_set.assign(**yield_codes).mul(regional_price_set).sum(1)
    return gpw


def main(crude, destination, refinery_volume = None, lvn_gasolinepool = None, kero_gasolinepool = None, reformer_capacity = None, fcc_capacity = None):                                       
    refinery_volume, reformer_input, kero_input, hvn_input, vgo_input, lvn_gasolinepool, kero_gasolinepool = cdu(crude, refinery_volume, lvn_gasolinepool, kero_gasolinepool)
    reformer_output = reformer(refinery_volume, reformer_input, kero_input, hvn_input, reformer_capacity)
    fcc_output = fcc(refinery_volume, vgo_input, fcc_capacity)
    diluent = fo_blend(crude, refinery_volume, fcc_output)
    yields = calculate_yields(crude, refinery_volume, reformer_output, fcc_output, lvn_gasolinepool, kero_gasolinepool, diluent)
    gpw = calculate_margin(yields, destination, ports, sub_to_ws)
    return gpw

#main(crude)
    





#main(crude)

#final_yields['lpg']['yield'] = 2    
    
    
    

    
    