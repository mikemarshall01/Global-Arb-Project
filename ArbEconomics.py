# -*- coding: utf-8 -*-
"""
Created on Sat Mar 3 18:09:50 2018

@author: mima

Need flat rate table
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from ImportData import ImportData 
from pandas.tseries.offsets import BDay


def arb(crude,destination,arb_data): 

    #crude = 'Urals Nth'
    #destination = 'Augusta'
       
    assay = arb_data.assay
    ws = arb_data.ws
    ports = arb_data.ports
    total = arb_data.total
    rate_data = arb_data.rate_data
    sub_to_ws = arb_data.sub_to_ws
    df = arb_data.df
    basrah_ws_base = arb_data.basrah_ws_base
    crude_diffs = arb_data.crude_diffs
    forties_sulphur = arb_data.forties_sulphur
    forties_sulphur.index.year
    exceptions = arb_data.exceptions
    #print(time.process_time() - t)
    
    """create the dataframes for use later"""
    df_freight = pd.DataFrame(index=df.index)
    df_prices = pd.DataFrame(index=df.index)
    
    index_wti = [x.lower().strip() for x in ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI','WTI']]
    index_dtd = [x.lower().strip() for x in ['DATED BRENT', 'DATED','N.SEA DATED','BTC Dated', 'MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','    GHANA DATED']]
    index_dub = [x.lower().strip() for x in ['DUBAI','DUBAI M2','OMAN/DUBAI']]
    crudes_diff_against_osp = ['Basrah Light','Barsah Heavy']
    
    """Declare the main prices that will be used in order to use shorthand notation"""
    dtd = total['PCAAS00']
    dub = total['AAVMR00']
    wtim1 = total['CLc1'] 
    wtim2 = total['CLc2']
    brentm1 = total['LCOc1']
    brentm2 = total['LCOc2']
    wti_cma_m1 = total['AAVSN00']
    cfd1 = total['PCAKG00']
    cfd2 = total['AAGLV00']
    cfd3 = total['PCAKE00']
    cfd4 = total['PCAKG00']
    cfd5 = total['AAGLU00']
    cfd6 = total['AAGLV00']
    cfd7 = total['AALCZ00']
    cfd8 = total['AALDA00']
    wti_br_m1 = wtim1 - brentm1
    wtim1_m2 = wtim1-wtim2
    brentm1_m2 = brentm1 - brentm2
    efpm2 = total['AAGVX00']
    efs2 = total['AAEBS00']
    mars_wti2 = total['AAKTH00']
    dfl_m1 = total['AAEAA00']

    days = 5
    loadport = assay[crude]['LoadPort']   
    sub_region = ports[ports['PortName'] == loadport]['SubRegionName'].map(sub_to_ws['WS_Region']).to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
    sub_region_2 = ports[ports['PortName'] == destination]['SubRegionName'].map(sub_to_ws['WS_Region']).to_string(index = False)
    discharge_price_region = ports[ports['PortName'] == destination]['SubRegionName'].map(sub_to_ws['Price_Set']).to_string(index = False)

    expiry_condition = df.index < pd.to_datetime(df.WTI_Expiry)
    cfd_condition = cfd1 > cfd2

    """Need to handle the one month forward OSP concept, so here, take the dataframe for the exceptions above, condense to monhtly values which wont
    change the value as same for each day, shift that forward then re-expand"""
    if assay[crude]['Code'] == 'multiple':         
        diff = total[exceptions[(exceptions['Name']==crude)&(exceptions['Destination']==discharge_price_region)]['Code'].iat[0]]
        crude_vs = exceptions[(exceptions['Name']==crude)&(exceptions['Destination']==discharge_price_region)]['Index'].iat[0].lower().strip()
    else:    
        diff = total[assay[crude]['Code']]
        crude_vs = assay[crude]['Index'].lower().strip()       
        
    """This is to make sure we use the CIF port if within the loaidng region, or switch to FOB loading port and FOB price if not
    The Urals clause is in there because we want to use the baltic to NWE quote for some reason not in the pecking order sheet"""
    if (assay[crude]['Basis'] == 'CIF') & (sub_region_2 != sub_region) & (crude == 'Urals Nth'):
        loadport = assay[crude]['FOBLoadPort']
        sub_region = ports[ports['PortName'] == loadport]['SubRegionName'].to_string(index = False)
    elif (assay[crude]['Basis'] == 'CIF') & (sub_region_2 != sub_region):
        loadport = assay[crude]['FOBLoadPort']
        sub_region = ports[ports['PortName'] == loadport]['SubRegionName'].map(sub_to_ws['WS_Region']).to_string(index = False)       
    else:
        pass 
    
    def construct_freight():
        def calculate_flat_rate():
            """create the flat rates table for the rates calculations and column creation"""    
            flat_rate_table = rate_data.loc[(rate_data['LoadPort'] == loadport)&
                      (rate_data['DischargePort'] == destination)]
            
            def calculate_flat_rates(x):
                return float(flat_rate_table.loc[flat_rate_table['Year'].astype(int) == x, 'Rate'])
            
            """Vectorising the function amkes it applicable over an array - before had to use pandas which was element wise application - i.e. SLOW"""
            v_calculate_flat_rates = np.vectorize(calculate_flat_rates)
            df_freight['Rate'] = np.apply_along_axis(v_calculate_flat_rates,0,np.array(df.index.year))
            if ports[ports['PortName'] == destination]['Country'].iat[0] == 'South Korea':
                flat_rate_table = rate_data.loc[(rate_data['LoadPort'] == 'Ruwais')&
                      (rate_data['DischargePort'] == 'Singapore')]
                v_calculate_flat_rates = np.vectorize(calculate_flat_rates)
                df_freight['Murban_Sing_Flat'] = np.apply_along_axis(v_calculate_flat_rates,0,np.array(df.index.year))
                
            return df_freight
        
        def calculate_port_costs():
            """These are for the odd costs, tax rebates, etc"""
            df_freight['Costs'] = 0
            
            # This is the export cost out of Houston
            if sub_region in (['US GULF (padd 3)']):
                df_freight['Houston_Load_Costs'] = np.where(df_freight.index > dt(2018,2,28),0.09,0)
                df_freight['Costs'] += df_freight['Houston_Load_Costs']
            
            # Port costs to discharge in Rotterdam               
            if destination == 'Rotterdam':
                df_freight['Rott_Discharge_Costs'] = 0.15
                df_freight['Costs'] += df_freight['Rott_Discharge_Costs']
            
            # Port costs to discharge in Houston
            if destination == 'Houston':
                df_freight['Hous_Discharge_Costs'] = 0.25
                df_freight['Costs'] += df_freight['Hous_Discharge_Costs']
                
            if loadport == 'Basrah':
                df_freight['Basrah_Costs'] = 0.76
                df_freight['Costs'] += df_freight['Basrah_Costs']
                
            if loadport == 'Ras Tanura':
                df_freight['Saudi_Costs'] = 0.66
                df_freight['Costs'] += df_freight['Saudi_Costs']
         
            return df_freight                

        def freight_and_quality_exceptions():
            if crude in ('Forties'):
                df_freight['Buzzard_Content'] = forties_sulphur['BuzzardContent']
                df_freight['Implied_Sulphur'] = df_freight['Buzzard_Content'] * 0.012 + 0.003
                df_freight['De-Escalator_Threshold'] = np.round(df_freight['Implied_Sulphur'], 3)
                df_freight['De-Escalator_Counts'] = np.minimum(0, 6-df_freight['Implied_Sulphur']*1000)
                df_freight['Platts_De_Esc'] = total['AAUXL00']
                df_freight['Forties_Margin_Impact'] = df_freight['Platts_De_Esc'] * df_freight['De-Escalator_Counts'] * -1
                df_freight['Costs'] += df_freight['Forties_Margin_Impact']
            
            if crude in ('Basrah Light','Basrah Heavy'):
                """This handles the freight escalation calculation from Iraq - the base is sent by SOMO, and table is in databse / excel wb"""
                monthly_averages = total['PFAOH00'].asfreq(BDay()).resample('BMS').mean() # resampled so we have the business month start, corrects averaging error if cma
                func_ma_on_days = lambda x: (monthly_averages.loc[(monthly_averages.index.month == x.month)&(monthly_averages.index.year == x.year)]).iat[0]
                
                """Create funcs to handle basrah base and flat rate values, apply over df and calc esclator"""
                func_ws_base = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['SOMO_WS']).iat[0]
                func_fr = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['SOMO_FlatRate']).iat[0]
                func_bhapi = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['BasrahHeavyAPI']).iat[0]                
                func_blapi = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['BasrahLightAPI']).iat[0]                
                df_freight['Date'] = df_freight.index
                df_freight['WS Month Avg'] = df_freight['Date'].apply(func_ma_on_days)
                df_freight['SOMO Base WS'] = df_freight['Date'].apply(func_ws_base)
                # We have to apply the corrcetion here after SOMO dropped their base rate earlier this year - assumption
                # only valid for 2018
                df_freight['SOMO Base WS'].iloc[(df_freight.index >= dt(2018,4,1))&(df_freight.index <= dt(2018,12,31))] = 25               
                df_freight['Base_FR_for_esc'] = df_freight['Date'].apply(func_fr)
                
                if crude == 'Basrah Light':
                    df_freight['API Esc'] = df_freight['Date'].apply(func_blapi)
                else:
                    df_freight['API Esc'] = df_freight['Date'].apply(func_bhapi)               
                
                
                df_freight['WS for Esc'] = (df_freight['WS Month Avg'] - df_freight['SOMO Base WS']) * df_freight['Base_FR_for_esc'] / 7.3 / 100
                df_freight.drop(['Date'], axis = 1, inplace=True)

             # South Korean particulars
            if ports[ports['PortName'] == destination]['Country'].iat[0] == 'South Korea':
                # Freight rebate on imported crudes
                df_freight['Murban_Freight_Comp'] = total['PFAOC00'] / 100 * df_freight['Murban_Sing_Flat'] / 7.66 #Murban density conversion
                df_freight['UKC-Yosu_VLCC'] = total['AASLA00'] * 1000000 / 2000000
                df_freight['Freight_Rebate'] = np.maximum(df_freight['UKC-Yosu_VLCC'] - df_freight['Murban_Freight_Comp'], 0.6)
                df_freight['Costs'] -= df_freight['Freight_Rebate']
                
                # Tax rebate on crudes out of Europe
                if ports[ports['PortName'] == loadport]['RegionName'].iat[0] in (['NW EUROPE','MED']):
                    df_freight['FTA_Tax_Rebate'] = 0.006 * total['LCOc1']
                    df_freight['Costs'] -= df_freight['FTA_Tax_Rebate']
                
                # Tax rebate on crudes out of the US
                if ports[ports['PortName'] == loadport]['RegionName'].iat[0] in (['N AMERICA']):
                    df_freight['FTA_Tax_Rebate'] = 0.005 * total['CLc1']
                    df_freight['Costs'] -= df_freight['FTA_Tax_Rebate']
                
                # Costs ascociated with lifting CPC based on delays
                if crude == 'CPC Blend':
                    df_freight['TS_Delays'] = np.maximum(total['AAWIL00'] + total['AAWIK00'] - 2,0)
                    df_freight['TS_Demur'] = total['AAPED00']
                    df_freight['TS_Demur_Costs'] = df_freight['TS_Delays'].mul(df_freight['TS_Demur'])/130
                    df_freight['Costs'] += df_freight['TS_Demur_Costs']
                
                # Costs ascociated with lifting Urals, actually a rebate as giving back port costs that are included in CIF price
                if crude in (['Urals Nth', 'Urals Med']):
                    df_freight['Urals_Cif_Rebate'] = 0.11
                    df_freight['Costs'] -= df_freight['Urals_Discharge_Costs']
                    
                if crude == 'Forties':
                    df_freight['Forties_Mkt_Discount'] = 0.5
                    df_freight['Costs'] -= df_freight['Forties_Mkt_Discount']
            else:
                 pass
            
            return df_freight
        
        def calculate_freight():
            
            #crude = 'Urals Nth'
            #destination = 'Augusta'
            
            if (assay[crude]['Basis'] == 'CIF')& (sub_region_2 != sub_region):
                """We need to essentially do the calculate freight function again, but rename the 'Rate' column to 'Onward_Rate"""
                loadport = assay[crude]['FOBLoadPort']  
                cif_destination = assay[crude]['LoadPort']                
                flat_rate_table = rate_data.loc[(rate_data['LoadPort'] == loadport)&
                      (rate_data['DischargePort'] == cif_destination)]
            
                def calculate_flat_rates(x):
                    return float(flat_rate_table.loc[flat_rate_table['Year'].astype(int) == x, 'Rate'])
                
                """Vectorising the function amkes it applicable over an array - before had to use pandas which was element wise application - i.e. SLOW"""
                v_calculate_flat_rates = np.vectorize(calculate_flat_rates)
                df_freight['Onward_Rate'] = np.apply_along_axis(v_calculate_flat_rates,0,np.array(df.index.year))
                
                """This is becaue we have black sea / baltics to UK CONT and MED but not to to other regions, so program will use freight where avliable"""
                if (crude == 'Urals Nth' and destination == 'Augusta') or (crude == 'Urals Med' and destination == 'Rotterdam'):
                    sub_region_onward = ports[ports['PortName'] == loadport]['SubRegionName'].to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
                else:    
                    sub_region_onward = ports[ports['PortName'] == loadport]['SubRegionName'].map(sub_to_ws['WS_Region']).to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
                sub_region_onward_2 = ports[ports['PortName'] == cif_destination]['SubRegionName'].map(sub_to_ws['WS_Region']).to_string(index = False)               
                ws_codes = ws[(ws['Origin'] == sub_region_onward)&(ws['Destination'] == sub_region_onward_2)]
                
                df_freight['Onward_Costs'] = 0
                if cif_destination == 'Rotterdam':
                    df_freight['Onward_Rott_Discharge_Costs'] = 0.15
                    df_freight['Onward_Costs'] -= df_freight['Onward_Rott_Discharge_Costs']
            
                onward_vessel_size = []
                for i in list(ws_codes['Code']):
                    #i = 'TDADP00'
                    onward_size = ws_codes[ws_codes['Code'] == i]['Size'].iat[0][0:3] + str('_Onward')
                    onward_vessel_size.append(onward_size)
                    onward_name = ws_codes[ws_codes['Code'] == i]['Name'].iat[0] + str('_Onward')
                    if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
                        df_freight[onward_name] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000
                        df_freight[onward_size] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['bbls'].values * 1000) + df_freight['Costs']
                        df_freight.drop(['Rate'], axis=1)
                    else:
                        df_freight[onward_name] = total[i]
                        df_freight[onward_size] = total[i] / 100 * df_freight['Onward_Rate'] / assay[crude]['Conversion'] - df_freight['Onward_Costs']

            """This finds the correct worldscale rate and adjusts if it is lumpsum"""
            ws_codes = ws[(ws['Origin'] == sub_region)&(ws['Destination'] == sub_region_2)]
            
            vessel_size = []
            
            for i in list(ws_codes['Code']):
                #i = 'PFAGN10'
                size = ws_codes[ws_codes['Code'] == i]['Size'].iat[0]
                vessel_size.append(size)
                name = ws_codes[ws_codes['Code'] == i]['Name'].iat[0]               
                grp_name = "Worldscale_" + name[:1]
                if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
                    df_freight[name] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000
                    df_freight[size] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['bbls'].values * 1000) + df_freight['Costs']
                    df_freight.drop(['Rate'], axis=1)                  
                elif crude in (['Hibernia','Terra Nova']):
                    df_freight[name] = total[i] / 1.25
                    df_freight[size] = total[i] / 1.25 / 100 * df_freight['Rate'] / assay[crude]['Conversion'] + df_freight['Costs']
                    df_freight[grp_name] = df_freight[name]
                else:
                    df_freight[name] = total[i]
                    df_freight[size] = total[i] / 100 * df_freight['Rate'] / assay[crude]['Conversion'] + df_freight['Costs']
                    df_freight[grp_name] = df_freight[name]

            
            if 'WS for Esc' in df_freight.columns.values:
                for i in vessel_size:
                    df_freight['Base_freight_'+str(i)[0:2]] = df_freight[i]
                    df_freight[i] = df_freight[i] - df_freight['WS for Esc'] + df_freight['API Esc']
                               
            if (assay[crude]['Basis'] == 'CIF')& (sub_region_2 != sub_region):
                for i in list(zip(vessel_size,onward_vessel_size)):
                    df_freight['FOB to Dest'] = df_freight[i[0]] 
                    df_freight[i[0]] = df_freight[i[0]] - df_freight[i[1]]
                                
            return df_freight
        
        calculate_flat_rate()
        calculate_port_costs()
        freight_and_quality_exceptions()       
        calculate_freight()

        return df_freight

    def convert_prices():
        if crude in crudes_diff_against_osp:
            df_prices['OSP'] = diff
            df_prices['Diff to OSP'] = crude_diffs[crude]
            df_prices['diff'] = diff + crude_diffs[crude]
        else:
            df_prices['diff'] = diff
        """depending on discharge, choose the appropriate index"""
        def convert_wti():
            df_prices['outright'] = wtim1
            if crude_vs in ['wti cma']:
                df_prices['wti_cma_m1'] = wti_cma_m1
                df_prices['wtim1'] = wtim1
                df_prices['vs_wti'] = diff + wti_cma_m1 - wtim1
            
            elif crude_vs in ['mars']:
                df_prices['mars_wti2'] = mars_wti2
                df_prices['vs_wti'] = diff + mars_wti2
     
            elif crude_vs in index_wti:
                df_prices['wtim1_m2'] = wtim1_m2
                df_prices['vs_wti'] = np.where(expiry_condition,
                         diff + wtim1_m2,
                         diff)
                
            elif crude_vs in index_dtd:
                cfd_condition = cfd4 > cfd8
                df_prices['cfd4'] = cfd4
                df_prices['cfd8'] = cfd8
                df_prices['efpm2'] = efpm2
                df_prices['wti_br_m1'] = wti_br_m1
                df_prices['vs_wti']  = np.where(cfd_condition,
                         diff + cfd8 +  efpm2 - (wti_br_m1),
                         diff + cfd4 +  efpm2 - (wti_br_m1))

            elif crude_vs in index_dub:
                """ This is because all the eastern crudes heading here have diffs against BWAVE"""
                pass
            else:
                df_prices['vs_wti'] = diff
            return df_prices
               
        def convert_dtd():
            df_prices['outright'] = dtd
# =============================================================================
#             conditions = [(expiry_condition & cfd_condition),
#                           (expiry_condition & np.invert(cfd_condition)),
#                           (np.invert(expiry_condition) & cfd_condition),
#                           (np.invert(expiry_condition) & np.invert(cfd_condition))]
#             choices = [(diff +  wtim1_m2 - (brentm2 - wtim2) - efpm2 - cfd2),
#                        (diff +  wtim1_m2 - (brentm2 - wtim2) - efpm2 - cfd1),
#                        (diff - (brentm2 - wtim2) - efpm2 - cfd2),
#                        (diff - (brentm2 - wtim2) - efpm2 - cfd1)]
# =============================================================================

            if crude_vs in ['wti cma']:
                """Here use cfd 8 on reccomendation of Andrea as by the time it loads only cfd wk 8 applicable"""
                df_prices['wti_cma_m1'] = wti_cma_m1
                df_prices['wtim1'] = wtim1
                df_prices['wti_br_m1'] = wti_br_m1
                df_prices['wtim1_m2'] = wtim1_m2
                df_prices['efpm2'] = efpm2
                df_prices['cfd8'] = cfd8
                df_prices['vs_dtd'] = diff + wti_cma_m1 - wtim1 + wtim1_m2 + wti_br_m1 - efpm2 - cfd8 
                
            elif crude_vs in ['bwave']:
                """Here use cfd 8 on reccomendation of Andrea as by the time it loads only cfd wk 8 applicable"""
                df_prices['dfl_m1'] = dfl_m1
                df_prices['vs_dtd'] = diff - dfl_m1 
            
            elif crude_vs in index_wti:
                df_prices['cfd8'] = cfd8
                df_prices['wtim1_m2'] = wtim1_m2
                df_prices['wti_br_m1'] = wti_br_m1
                df_prices['efpm2'] = efpm2
                df_prices['vs_dtd'] = np.where(expiry_condition,
                         diff + wtim1_m2 + wti_br_m1 - efpm2 - cfd8,
                         diff + wti_br_m1 - efpm2 - cfd8)
            
            elif crude_vs in index_dtd:
                if crude in ('Basrah Light','Basrah Heavy'): 
                    df_prices['cfd3'] = cfd3
                    df_prices['cfd4'] = cfd4
                    df_prices['structure'] = ((cfd3 - cfd4)/7) * 7
                    df_prices['vs_dtd'] = df_prices['diff'] + df_prices['structure']
                else:
                    df_prices['cfd3'] = cfd3
                    df_prices['cfd5'] = cfd5
                    if sub_region == sub_region_2:
                        df_prices['vs_dtd'] = diff
                    elif sub_region == 'WAF':
                        df_prices['structure'] = ((cfd3 - cfd5)/14) * days
                        df_prices['vs_dtd'] = diff + df_prices['structure']
                    else:
                        df_prices['structure'] = ((cfd3 - cfd5)/7) * days
                        df_prices['vs_dtd'] = diff + df_prices['structure']
            else:
                df_prices['vs_dtd'] = diff
            return df_prices
    
        def convert_dub():
            df_prices['outright'] = dub
            if crude_vs in ['wti cma']:
                """Here use cfd 8 on reccomendation of Andrea as by the time it loads only cfd wk 8 applicable"""
                df_prices['wti_cma_m1'] = wti_cma_m1
                df_prices['wtim1'] = wtim1
                df_prices['wti_br_m1'] = wti_br_m1
                df_prices['wtim1_m2'] = wtim1_m2
                df_prices['efs2'] = efs2
                df_prices['vs_dub'] = diff + wti_cma_m1 - wtim1 + wtim1_m2 + wti_br_m1 + efs2 
                         
            elif crude_vs in index_wti:
                df_prices['wtim1_m2'] = wtim1_m2
                df_prices['wti_br_m1'] = wti_br_m1
                df_prices['brentm2'] = brentm2
                df_prices['efs2'] = efs2
                df_prices['vs_dub'] = np.where(expiry_condition,
                         diff + wtim1_m2 + wti_br_m1 + efs2,
                         diff + wti_br_m1 + efs2)
            
            elif crude_vs in index_dtd:
                df_prices['cfd6'] = cfd6
                df_prices['efpm2'] = efpm2
                df_prices['efs2'] = efs2
                df_prices['vs_dub'] = diff + cfd6 +  efpm2 + efs2
                
            else:
                df_prices['vs_dub'] = diff
                
            return df_prices
        
        index_region = ports[ports['PortName'] == destination]['SubRegionName'].map(sub_to_ws['local_index']).to_string(index = False)
        func_list = {'WTI':convert_wti, 'DATED':convert_dtd, 'OMAN/DUBAI':convert_dub}
        [f() for index, f in func_list.items() if index == index_region][0]
        return df_prices
    
    try:
        df_freight = construct_freight()
    except Exception as e: print(e), print('df_freight')    
    
    try:
        df_prices = convert_prices()
    except Exception as e: print(e), print('df_prices') 
    
    temp = pd.concat([df_prices,df_freight], axis=1)
    price_index = [price_index for price_index in df_prices.columns if 'vs_' in price_index][0]
    freight_list = [freight for freight in df_freight.columns if 'max' in freight or 'VLCC' in freight]
    
    try:
        
        for k in freight_list:
            try:
                name = str(k[:4]) + str('_landed_') + str(price_index)
            except Exception as e: print('name fails') 
            if destination == loadport:
                temp[name] = df_prices[price_index]
            else:
                try:
                    temp[name] = df_prices[price_index].add(df_freight[k])
                except Exception as e: print('temp fails')    
    except Exception as e: print('check')

    return temp

"""This is for debugging"""
if __name__ == "__main__":
    crude = 'AZERI'
    destination = 'Singapore'
    t5 = time.process_time()
    arb_data = ImportData()
    test_results = arb(crude,destination,arb_data)
    print("ArbData created successfully: Time was {}".format(time.process_time() - t5))
    


    
   