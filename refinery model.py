# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:49:51 2018

@author: mima
"""
import numpy as np 
import pandas as pd

# Get the Data
raw_assay = pd.read_excel('L://TRADING//ANALYSIS//GLOBAL//Arb Models//GPW model RA.xlsm', sheetname = 'Upload_Test', header = 0, index_col = 'Database_Name')

# with the crude name set as the index, we use pd.to_dict to set a dictioanry of dictionaries
assay = raw_assay.to_dict('index')


### INPUT CRUDE ###
crude = 'Brent'

refinery_volume = 200

### Fractions for gaosline pool
lvn_gasolinepool = 0.12
kero_gasolinepool = 0.15

# REFORMER
reformer_capacity = 42
reformer_output = {}
reformer_assay = {
        'c3_c4': {'yield': 0.05},
        'h2': {'yield': 0.0},
        'btx': {'yield': 0.0},
        'gasoline': {'yield': 0.95}
        }

utilised_ref_cap = reformer_capacity * 0.97
hvn_input = assay[crude]['HVN'] * refinery_volume
kero_input = assay[crude]['KERO'] * kero_gasolinepool * refinery_volume
reformer_input = hvn_input + kero_input
reformer_volume = np.minimum(utilised_ref_cap, reformer_input)
surplus_for_jet = np.maximum(np.minimum(reformer_input - utilised_ref_cap, kero_input), 0)
surplus_for_naphtha = np.maximum(hvn_input - utilised_ref_cap, 0)

reformer_output['c3_c4'] = reformer_assay['c3_c4']['yield'] * reformer_volume
reformer_output['h2'] = reformer_assay['h2']['yield'] * reformer_volume
reformer_output['btx'] = reformer_assay['btx']['yield'] * reformer_volume
reformer_output['gasoline'] = reformer_assay['gasoline']['yield'] * reformer_volume



# FCC    
fcc_capacity = 48
fcc_output = {}
fcc_assay = {
        'c3_c4': {'yield': 0.02},
        'c3__': {'yield': 0.02},
        'lco': {'yield': 0.20},
        'clo': {'yield': 0.20},
        'gasoline': {'yield': 0.56}
        }

utilised_fcc_cap = fcc_capacity * 0.97 # this is 97% ulitilisation of total FCC capacity
vgo_available = assay[crude]['VGO'] * refinery_volume
fcc_volume = np.minimum(utilised_fcc_cap, vgo_available)
surplus_vgo = np.maximum(vgo_available - utilised_fcc_cap, 0)

fcc_output['propane'] = fcc_assay['c3_c4']['yield'] * fcc_volume / 2
fcc_output['butane'] = fcc_assay['c3_c4']['yield'] * fcc_volume / 2
fcc_output['c3__'] = fcc_assay['c3__']['yield'] * fcc_volume
fcc_output['lco']  = fcc_assay['lco']['yield'] * fcc_volume
fcc_output['clo']  = fcc_assay['clo']['yield'] * fcc_volume
fcc_output['gasoline'] = fcc_assay['gasoline']['yield'] * fcc_volume
    


# FO BLENDING

### RESIDUE INPUT ###
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
lco_for_blending = 0
diluent_used = 0
x = [diluent_used]

diluent = {
        'volume': surplus_vgo + fcc_output['clo'] + imported_vgo,
        'sulphur': 0.001,
        'density': 0.897,
        'viscosity': 3
        }


cons = ({'type': 'ineq', 'fun': lambda x: x[0] - diluent['volume'] },
         {'type': 'ineq', 'fn': lambda x})




diluent['sulphurW'] = diluent['volume'] * diluent['sulphur'] * diluent['density'] * 1000 / 6.29
diluent['viscosity_index'] = 23.1 + np.log10(np.log10(diluent['viscosity'] + 0.8)) * 33.47

leftover_diluent = diluent['volume'] - diluent_used # need to have an optimizer function of some sort

diluent['sulphur_per_bbl'] = diluent_used * diluent['density'] * 1000 / 6.29 * diluent['sulphur']
diluent['weight'] = diluent_used * diluent['density'] * 1000 / 6.29
residue['sulphur_per_bbl'] = residue['volume'] * residue['density'] * 1000 / 6.29 * residue['sulphur']
residue['weight'] = residue['volume']  * residue['density'] *1000 / 6.29

combined_fo_sulphur = diluent['sulphur_per_bbl'] + residue['sulphur_per_bbl'] 
combined_fo_sulphur_weight = diluent['weight'] + residue['weight']

target_viscosity = 10 ** (10 ** (((diluent_used / (diluent_used + residue['volume']) * diluent['viscosity_index']
                                    + residue['volume'] / (diluent_used + residue['volume']) * residue['viscosity_index'])-23.1)/33.47)) -0.8
target_sulphur = combined_fo_sulphur / combined_fo_sulphur_weight
                                    
                                    
                                    
final_yields = {}
                      
final_yields['lpg'] = (assay[crude]['LPG'] * refinery_volume + reformer_output['c3_c4'] + fcc_output['propane'] + fcc_output['butane'] + fcc_output['c3__']) / refinery_volume
final_yields['naphtha'] = (assay[crude]['LVN'] * refinery_volume * (1-lvn_gasolinepool) + surplus_for_naphtha) / refinery_volume
final_yields['btx'] = (reformer_output['btx'])/ refinery_volume
final_yields['gasoline'] = (reformer_output['gasoline'] + fcc_output['gasoline']) / refinery_volume
final_yields['kero'] = (assay[crude]['KERO'] * (1-kero_gasolinepool) * refinery_volume + surplus_for_jet) / refinery_volume
final_yields['ulsd'] = ((assay[crude]['LGO'] + assay[crude]['HGO']) * refinery_volume) / refinery_volume
final_yields['gasoil'] = (fcc_output['lco'] - lco_for_blending + leftover_diluent) / refinery_volume
final_yields['lsfo'] = (diluent_used + assay[crude]['RESIDUE'] * refinery_volume) / refinery_volume
final_yields['f_l'] = (final_yields['lpg'] + final_yields['naphtha'] + final_yields['btx'] + final_yields['gasoline'] + final_yields['kero']+ final_yields['ulsd'] + final_yields['gasoil'] + final_yields['lsfo'] - 1) 

print(final_yields)                                        
    
    
    
    
    
    
    

    
    