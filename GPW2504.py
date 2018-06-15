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
from ArbEcons2504 import import_data

#crude = 'Azeri'
#destination = 'Rotterdam'
#assay, ws, ports, total, rate_data, sub_to_ws, df = import_data()


np.seterr(divide='ignore', invalid='ignore')

def gpw_calculation(crude, destination, assay, ws, ports, total, rate_data, sub_to_ws, df, basrah_ws_base, crude_diffs, forties_sulphur, exceptions, crudes_to_shift, **refinery_config):

    def standard_ref(crude, refinery_volume, reformer_capacity, fcc_capacity, coker_capacity, lvn_gasolinepool, kero_gasolinepool):
        
        def cdu():
            ### Fractions for gaosline pool       
            hvn_input = assay[crude]['HVN'] * refinery_volume
            kero_input = assay[crude]['KERO'] * kero_gasolinepool * refinery_volume
            lvn_input = assay[crude]['LVN'] * lvn_gasolinepool * refinery_volume
            reformer_input = hvn_input + kero_input + lvn_input
            vgo_input = assay[crude]['VGO'] * refinery_volume
            coker_input = assay[crude]['RESIDUE'] * refinery_volume
            
            return refinery_volume, reformer_input, kero_input, hvn_input, vgo_input, lvn_gasolinepool, kero_gasolinepool, coker_input
        
        def reformer():
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
            
            """if the reformer input is greater than the capacity, say 50kbd vs 48kbd, this means 2kbd is surplus. This 2kbs is then compared to the kero input
            which is part of the feed for the reformer. We can't have more surplus jet than the volume put in so we compare the additional amount to the
            kero_input. If the kero input is larger than the surplus, we assume all additional goes into the jet pool, however, if the ref_input vs the cap is higher than
            the kero input, then the minimum clause says just take the kero input as it will be the smaller of the two and then the max means we take that as the volume 
            of the surplus"""
            
            reformer_output['surplus_for_jet'] = np.maximum(np.minimum(reformer_input - utilised_ref_cap, kero_input), 0)
            reformer_output['surplus_for_naphtha'] = np.maximum(hvn_input - utilised_ref_cap, 0)
            reformer_output['propane'] = reformer_assay_standard['c3']['yield'] * reformer_volume 
            reformer_output['butane'] = reformer_assay_standard['c4']['yield'] * reformer_volume 
            reformer_output['h2'] = reformer_assay_standard['h2']['yield'] * reformer_volume
            reformer_output['btx'] = reformer_assay_standard['btx']['yield'] * reformer_volume
            reformer_output['gasoline'] = reformer_assay_standard['gasoline']['yield'] * reformer_volume
            
            return reformer_output
        
        def coker():
            coker_output = {}
            coker_assay_standard = {
                'c3': {'yield': 0.015},
                'c4': {'yield': 0.015},
                'lvn': {'yield': 0.09},
                'hvn': {'yield': 0.11},
                'lgo': {'yield': 0.15},
                'hgo': {'yield': 0.35}
                }
            utilised_coker_cap = coker_capacity * 0.97
            coker_volume = np.minimum(utilised_coker_cap, coker_input)
            
            coker_output['surplus_resid'] = np.maximum(coker_input - utilised_coker_cap,0)
            coker_output['propane'] = coker_assay_standard['c3']['yield'] * coker_volume
            coker_output['butane'] = coker_assay_standard['c4']['yield'] * coker_volume
            coker_output['lvn'] = coker_assay_standard['lvn']['yield'] * coker_volume
            coker_output['hvn'] = coker_assay_standard['hvn']['yield'] * coker_volume
            coker_output['lgo'] = coker_assay_standard['lgo']['yield'] * coker_volume
            coker_output['hgo'] = coker_assay_standard['hgo']['yield'] * coker_volume
            
            return coker_output

                
        def fcc():
            fcc_output = {}
            
            fcc_assay_standard = {
                'c3': {'yield': 0.01},
                'c4': {'yield': 0.01},
                'c3__': {'yield': 0.02},
                'lco': {'yield': 0.20},
                'clo': {'yield': 0.20},
                'gasoline': {'yield': 0.56}
                }
            
            fcc_assay_usgc = {
                'c3': {'yield': 0.015},
                'c4': {'yield': 0.015},
                'lco': {'yield': 0.12},
                'clo': {'yield': 0.08},
                'gasoline': {'yield': 0.75}
                }
            
            utilised_fcc_cap = fcc_capacity * 0.97 # this is 97% ulitilisation of total FCC capacity
            fcc_volume = np.minimum(utilised_fcc_cap, vgo_input + coker_output['hgo'])
            
            if  ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[3]).iat[0] is 'HOUSTON':
                fcc_assay = fcc_assay_usgc
            else:
                fcc_assay = fcc_assay_standard
            
            
            fcc_output['surplus_vgo'] = np.maximum(vgo_input + coker_output['hgo'] - utilised_fcc_cap, 0)
            fcc_output['propane'] = fcc_assay['c3']['yield'] * fcc_volume
            fcc_output['butane'] = fcc_assay['c4']['yield'] * fcc_volume
            fcc_output['c3__'] = fcc_assay['c3__']['yield'] * fcc_volume
            fcc_output['lco']  = fcc_assay['lco']['yield'] * fcc_volume
            fcc_output['clo']  = fcc_assay['clo']['yield'] * fcc_volume
            fcc_output['gasoline'] = fcc_assay['gasoline']['yield'] * fcc_volume
            
            return fcc_output
        

        def alkylation_unit():
            pass
        
        def hydrocracker():
            pass
        
        def fo_blend():
            residue = {}
            residue['volume'] = coker_output['surplus_resid']
            
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
        
        def calculate_yields():
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
                    + reformer_output['surplus_for_naphtha'] + coker_output['lvn']
                    ) / refinery_volume
                                                        
            #yields['btx'] = (reformer_output['btx']) / refinery_volume
            
            yields['gasoline'] = (reformer_output['gasoline'] + fcc_output['gasoline'] + coker_output['hvn']) / refinery_volume
            yields['kero'] = (assay[crude]['KERO'] * (1-kero_gasolinepool) * refinery_volume + reformer_output['surplus_for_jet']) / refinery_volume
            yields['ulsd'] = ((assay[crude]['LGO'] + assay[crude]['HGO']) * refinery_volume + coker_output['lgo']) / refinery_volume 
            yields['gasoil'] = (fcc_output['lco'] - diluent['lco_for_blending'] + diluent['surplus_after_blend']) / refinery_volume
            yields['lsfo'] = (diluent['volume_used'] + assay[crude]['RESIDUE'] * refinery_volume) / refinery_volume
    
            return yields
        
        refinery_volume, reformer_input, kero_input, hvn_input, vgo_input, lvn_gasolinepool, kero_gasolinepool, coker_input = cdu()
        reformer_output = reformer()
        coker_output = coker()
        fcc_output = fcc()
        diluent = fo_blend()
        yields = calculate_yields()
        return yields
    
    def calculate_margin(yields):
    
        """The below gets us our list of prroduct price codes"""
        #[price_codes[pricing_centre][i]['Code'] for i,j in yields.items()]  
        
        #{price_codes[pricing_centre][i]['Code']: [price_codes[pricing_centre][i]['Conversion'] for (i, j) in yields.items()}
    
        yield_codes = {}
        for i, j in yields.items():
            yield_codes[price_codes[pricing_centre][i]['Code']] = j
        
        yield_names = {}
        for i, j in yields.items():
            yield_names[price_codes[pricing_centre][i]['Code']] = i
    
        conversion_codes = {}
        for i, j in yields.items():
            conversion_codes[price_codes[pricing_centre][i]['Code']] = price_codes[pricing_centre][i]['Conversion']
        
        regional_price_set = total[list(yield_codes.keys())].dropna(axis=0)
        temp_df = regional_price_set.assign(**conversion_codes)
        gpw = regional_price_set.div(temp_df)
        
        temp_df = regional_price_set.assign(**yield_codes)
        gpw = gpw.mul(temp_df)
        gpw.rename(columns=yield_names, inplace=True)
        
        gpw['GPW'] = gpw.sum(1)
        return gpw
               
    """this is to create the look up neccessary for generating the table"""
    price_codes = {'HOUSTON':{'propane': {'Code':'PMAAY00', 'Conversion':(100/42)},
                      'butane':{'Code':'PMAAI00', 'Conversion':(100/42)},
                      'naphtha':{'Code':'AAXJP00', 'Conversion':(100/42)},
                      'gasoline':{'Code':'AAVKS00', 'Conversion':(100/42)},
                      'kero':{'Code':'PJABP00', 'Conversion':(100/42)},
                      'ulsd':{'Code':'AATGX00', 'Conversion':(100/42)},
                      'gasoil':{'Code':'POAED00', 'Conversion':(100/42)},
                      'lsfo':{'Code':'PUAAO00', 'Conversion':1},
                      'base_blend':'BaseBlendUSGC'},
        'ROTTERDAM':{'propane': {'Code':'PMABA00', 'Conversion':11},
                      'butane':{'Code':'PMAAK00', 'Conversion':11},
                      'naphtha':{'Code':'PAAAM00', 'Conversion':8.9},
                      'gasoline':{'Code':'AAQZV00', 'Conversion':8.34},
                      'kero':{'Code':'PJABA00', 'Conversion':7.87},
                      'ulsd':{'Code':'AAJUS00', 'Conversion':7.45},
                      'gasoil':{'Code':'AAYWT00', 'Conversion':7.45},
                      'lsfo':{'Code':'PUAAP00', 'Conversion':6.32},
                      'base_blend':'BaseBlendNWE'},
        'AUGUSTA':{'propane': {'Code':'PMABC00', 'Conversion':11},
                      'butane':{'Code':'PMAAM00', 'Conversion':11},
                      'naphtha':{'Code':'PAAAH00', 'Conversion':8.9},
                      'gasoline':{'Code':'AAWZB00', 'Conversion':8.34},
                      'kero':{'Code':'AAZBN00', 'Conversion':7.87},
                      'ulsd':{'Code':'AAWYZ00', 'Conversion':7.45},
                      'gasoil':{'Code':'AAVJJ00', 'Conversion':7.45},
                      'lsfo':{'Code':'PUAAJ00', 'Conversion':6.32},
                      'base_blend':'BaseBlendMED'},
        'SINGAPORE':{'propane': {'Code':'AAJTQ00', 'Conversion':11},
                      'butane':{'Code':'AAJTT00', 'Conversion':11},
                      'naphtha':{'Code':'PAAAP00', 'Conversion':1},
                      'gasoline':{'Code':'PGAEY00', 'Conversion':1},
                      'kero':{'Code':'PJABF00', 'Conversion':1},
                      'ulsd':{'Code':'AAFEX00', 'Conversion':1},
                      'gasoil':{'Code':'AAFEX00', 'Conversion':1},
                      'lsfo':{'Code':'PUADV00', 'Conversion':6.32},
                      'base_blend':'BaseBlendASIA'}
                             }
                
    """Choose which set of products we use based on discharge locations"""   
    pricing_centre = ports[ports['Name'].str.lower() == destination.lower()]['Subregion'].map(sub_to_ws[3]).to_string(index = False) 
    
    refinery_volume = refinery_config['refinery_volume']
    reformer_capacity = refinery_config['reformer_capacity']
    fcc_capacity = refinery_config['fcc_capacity']
    coker_capacity = refinery_config['coker_capacity']
    lvn_gasolinepool = refinery_config['lvn_gasolinepool']
    kero_gasolinepool = refinery_config['kero_gasolinepool']
         
# =============================================================================
#     if refinery_volume is None:
#         refinery_volume = 200
#     if lvn_gasolinepool is None:
#         lvn_gasolinepool = 0.12
#     if kero_gasolinepool is None:
#         kero_gasolinepool = 0.15
#     if reformer_capacity is None:
#         reformer_capacity = refinery_volume*0.21
#     if fcc_capacity is None:
#             fcc_capacity = refinery_volume*0.24
# =============================================================================
    if coker_capacity is 0 and destination is 'Houston':
            coker_capacity = refinery_volume*0.10
            
# =============================================================================
#     refinery_config = {'refinery_volume':refinery_volume,
#         'reformer_capacity':reformer_capacity,
#         'fcc_capacity':fcc_capacity,
#         'coker_capacity':coker_capacity,
#         'lvn_gasolinepool':lvn_gasolinepool,
#         'kero_gasolinepool':kero_gasolinepool
#         }       
#     
# =============================================================================
    """This returns a dataframe with crack values and the sum of the GPW dependent on the ref config we have. 
    This is essentially saying to run the algo for the choice of crude through the refining configuration given. So for simple and complex,
    we need to pass alternative configurations"""
    
    base_yields = standard_ref(price_codes[pricing_centre]['base_blend'],refinery_volume, reformer_capacity, fcc_capacity, coker_capacity, lvn_gasolinepool, kero_gasolinepool)
    base_crude = calculate_margin(base_yields)
    
    
    crude_yields = standard_ref(crude, refinery_volume, reformer_capacity, fcc_capacity, coker_capacity, lvn_gasolinepool, kero_gasolinepool)
    crude_gpw = calculate_margin(crude_yields)
    
    crude_gpw['Base_GPW'] = base_crude['GPW']
    #crude_gpw['GPW_delta'] = crude_gpw['GPW'] - crude_gpw['Base_GPW']
    return crude_gpw


#crude_gpw = gpw_calculation(crude, destination, assay, ws, ports, total, rate_data, sub_to_ws, df)














