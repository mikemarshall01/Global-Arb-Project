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
from pandas.tseries.offsets import BDay


    
def import_data():
    t2 = time.process_time()
    #data = pd.ExcelFile('C://Users//mike_//Downloads//toydata1004.xlsx')
    #raw_rates = pd.ExcelFile('C://Users//mike_//Downloads//FlatRatesComplete.xlsx')
    
    data = pd.ExcelFile('C://Users//mima//Documents//toydata1004.xlsx')
    raw_rates = pd.ExcelFile('C://Users//mima//Documents//FlatRatesComplete.xlsx')
    
    trader_assessed = pd.ExcelFile('L://TRADING//ANALYSIS//GLOBAL//Arb Models//Pecking Order 2018.xlsm')
    
  
    assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
    ws = pd.read_excel(data, 'ws')
    expiry_table = pd.read_excel(data, 'expiry', index_col = 'Month')
    ports = pd.read_excel(data, 'ports')
    sub_to_ws = pd.read_excel(data, 'sub_to_ws', header = None)
    sub_to_ws = sub_to_ws.set_index([0]).to_dict()
    
    """table containing the basrah base worldscale that they fix their freight against"""
    basrah_ws_base = pd.read_excel(data, 'basrah_ws_base', index_col = 'YEAR')
    
    """Take in the crude prices and codes and convert to a dataframe.
    We need to take the first 2 rows of the prices with no headers as this will give us the cude name and the code ascociated
    Then transpose from rows to columns and rename the columns. This will be for later when we determine crude prices basis desired comaprison"""
    #prices_reference = (pd.read_excel(data, 'paper prices', header = None).iloc[0:2,1:]).transpose().rename(columns={0:'Name', 1: 'Code'})  
    
    """Merge the WS table with the prices table, slice df so 2016 onwards (Flat rates last date is 2015). 
    We don't drop rows now as dropping would be dependent on any nans in any column"""
    #total = prices.merge(ws_table, how = 'inner', left_index = True, right_index = True)
    #total = total.merge(paper_prices, how = 'inner', left_index = True, right_index = True)
    #total = total.iloc[total.index > dt(2015,12,31)]
    
    """this new total table generates all the prices in one place for us"""
    total = pd.read_excel(data, 'price_warehouse', header = 4).drop(['Timestamp'])
    total.index = pd.to_datetime(total.index)
    total.sort_index(inplace=True)
    total.fillna(method='ffill', inplace=True)
    total = total[total.index > dt(2015,1,1)]
    
    """We know there are some perculiarities in the data, such as the OSPs. So create this table here to handle. Found out need to shift the prices back a month but in order
    to identify which ones, needed the list of OSP crudes"""
    exceptions = {
            'Arab Extra Light':
                {'ROTTERDAM':{'Code':'AAIQQ00','Index':'BWAVE'},
                 'AUGUSTA':{'Code':'AAWQK00','Index':'BWAVE'},
                 'HOUSTON':{'Code':'AAIQZ00','Index':'WTI'},
                 'SINGAPORE':{'Code':'AAIQV00','Index':'OMAN/DUBAI'}},
            'Arab Light':
                {'ROTTERDAM':{'Code':'AAIQR00','Index':'BWAVE'},
                'AUGUSTA':{'Code':'AAWQL00','Index':'BWAVE'},
                'HOUSTON':{'Code':'AAIRA00','Index':'WTI'},
                'SINGAPORE':{'Code':'AAIQW00','Index':'OMAN/DUBAI'}},
            'Arab Medium':
                {'ROTTERDAM':{'Code':'AAIQS00','Index':'BWAVE'},
                 'AUGUSTA':{'Code':'AAWQM00','Index':'BWAVE'},
                 'HOUSTON':{'Code':'AAIRB00','Index':'WTI'},
                 'SINGAPORE':{'Code':'AAIQX00','Index':'OMAN/DUBAI'}},
            'Arab Heavy':
                {'ROTTERDAM':{'Code':'AAIQT00','Index':'BWAVE'},
                 'AUGUSTA':{'Code':'AAWQN00','Index':'BWAVE'},
                 'HOUSTON':{'Code':'AAIRC00','Index':'WTI'},
                 'SINGAPORE':{'Code':'AAIQY00','Index':'OMAN/DUBAI'}},
            'Basrah Light':
                {'ROTTERDAM':{'Code':'AAIPH00','Index':'Dated'},
                 'AUGUSTA':{'Code':'AAIPH00','Index':'Dated'},
                 'HOUSTON':{'Code':'AAIPG00','Index':'WTI'},
                 'SINGAPORE':{'Code':'AAIPE00','Index':'OMAN/DUBAI'}},
            'Basrah Heavy':
                {'ROTTERDAM':{'Code':'AAXUC00','Index':'Dated'},
                 'AUGUSTA':{'Code':'AAXUC00','Index':'Dated'},
                 'HOUSTON':{'Code':'AAXUE00','Index':'Mars'},
                 'SINGAPORE':{'Code':'AAXUA00','Index':'OMAN/DUBAI'}},
            'Iranian Heavy':
                {'ROTTERDAM':{'Code':'AAIPB00','Index':'BWAVE'},
                 'AUGUSTA':{'Code':'AAUCH00','Index':'BWAVE'},
                 #'Iranian Heavy':{'HOUSTON':{'Code':abcde,'Index':'WTI'}},
                'SINGAPORE':{'Code':'AAIOY00','Index':'OMAN/DUBAI'}},
            'Iranian Light':
                {'ROTTERDAM':{'Code':'AAIPA00','Index':'BWAVE'},
                 'AUGUSTA':{'Code':'AAUCJ00','Index':'BWAVE'},
                'SINGAPORE':{'Code':'AAIOX00','Index':'OMAN/DUBAI'}},
            'Forozan':
                {'ROTTERDAM':{'Code':'AAIPC00','Index':'BWAVE'},
                'AUGUSTA':{'Code':'AAUCF00','Index':'BWAVE'},
                'SINGAPORE':{'Code':'AAIOZ00','Index':'OMAN/DUBAI'}},
            'Isthmus':{'ROTTERDAM':{'Code':'AAIQC00','Index':'Dated'},
                'AUGUSTA':{'Code':'AAIQC00','Index':'Dated'},
                'HOUSTON':{'Code':'AAIPZ00','Index':'WTI'},
                'SINGAPORE':{'Code':'AAIQE00','Index':'OMAN/DUBAI'}},
            'Maya':{'ROTTERDAM':{'Code':'AAIQB00','Index':'Dated'},
                'AUGUSTA':{'Code':'AAIQB00','Index':'Dated'},
                'HOUSTON':{'Code':'AAIPY00','Index':'WTI'},
                'SINGAPORE':{'Code':'AAIQD00','Index':'OMAN/DUBAI'}}
            }
     
    crudes_to_shift = pd.DataFrame.from_dict({(crude,destination): exceptions[crude][destination] 
            for crude in exceptions.keys() 
            for destination in exceptions[crude].keys()}, 
            orient='index')
    
    """convert the dataseries to a list, then use setr to get the unique items, then convert back to a list"""   
    crudes_to_shift = list(set(list(crudes_to_shift['Code'])))
    
    """Fopr the crudes in the list, I want to resample the series at the month start so there is a common value for the start of each month,
    I then want shift these values by 1 backwards, in this case because we resampled, this automatically means shift abck one month,
    I then want to re-index the new dataframe to conform to where we are putting it back into, and finally I assign the total dataframe where the 
    column headers are equal to the crude list, the new shifted and filled forward values to make sure everything lines up"""
    total[crudes_to_shift] = total[crudes_to_shift].resample('MS').mean().shift(-1, freq='MS').reindex(total.index).fillna(method='ffill')  

    #total['AAXUC00']
    
    """This will help with the date error. Turn the index into a numpy array and then assign the value"""
    if total.index[-1] - total.index[-2] > pd.Timedelta(days=2):
        total.index.values[-1] = total.index[-2] + pd.Timedelta(days=1)


    """Clean the column hedaers so no white spcaes - use simple list comprehension and set headers equal to cleaned"""
    cleaned_column_headers = [i.strip() for i in total.columns.values]
    total.columns = cleaned_column_headers
    
    """The below was get rid of the row in the index that hax NaT against it and then expand to daily and fill backwards"""
    crude_diffs = pd.read_excel(trader_assessed, 'Crude Diffs Traders', header = 0)
    crude_diffs = crude_diffs.loc[pd.notnull(crude_diffs.index)]
    crude_diffs = crude_diffs.drop([name for name in crude_diffs.columns if 'Unnamed' in name], axis=1)

   
    #crude_diffs.index = crude_diffs.index.map(lambda x : x + 1*BDay())
    crude_diffs = crude_diffs.reindex(total.index).fillna(method='bfill').fillna(method='ffill')
    
    """Slice the crude diffs where the dates in the index are the same as the dates in the total dataframe"""
    #crude_diffs = crude_diffs[crude_diffs.index.isin(total.index)]
    crudes_diff_against_osp = ['Basrah Light','Basrah Heavy']
    codes_list = [x for x in crude_diffs.columns if x not in crudes_diff_against_osp]
    
    """Apply the values in crude diffs to the correct codes and dates in the total dataframe"""
    total.update(crude_diffs[codes_list])
    
    
        
    
    """We have to convert the prices that are in absolutes into a diff vs a local index, and if there are, set to zero.
    This is LOOP Sour"""
    total['AALSM01'].loc[total['AALSM01'] > 30] = total['AALSM01'].loc[total['AALSM01'] > 30] - total['CLc1']
    #total.loc[total.index.isin(crude_diffs.index), codes_list] = crude_diffs[codes_list]
    #total[codes_list]
    
    #total.update(crude_diffs[codes_list])
    """ Need this for the sulphur table"""
    forties_sulphur = pd.read_excel(trader_assessed, 'Forties de-esc', header = [22], parse_cols="H:I").set_index('week ending')
    forties_sulphur = forties_sulphur.loc[pd.notnull(forties_sulphur.index)]
    forties_sulphur = forties_sulphur.reindex(total.index).fillna(method='ffill')

    """Also need to adjust the cfds to take into account the inter month BFOE spread"""   
    cfd_list = ['PCAKA00','PCAKC00','PCAKE00','PCAKG00','AAGLU00','AAGLV00','AALCZ00','AALDA00']
    temp = total[cfd_list].sub(pd.Series(total['PCAAQ00'] - total['PCAAR00']), axis=0)
    temp = temp[temp.index > dt(2017,6,30)]
    total.loc[total.index.isin(temp.index), list(temp.columns)] = temp[list(temp.columns)]
    
    """This turns the 5 years of rate matricies into a table for use to reference - 12/04/2018"""    
    rates = []
    for x,y in enumerate([name.split()[2] for name in raw_rates.sheet_names]):
        f  = pd.read_excel(raw_rates, sheetname = x, header = None).iloc[1:47,1:]
        lplen = len(f.iloc[:,1])
        dplen = len(f.iloc[1,:])
        for j in range(1, dplen):
            for i in range(1,lplen):
                LoadPort = f.iloc[i,0]
                DischargePort = f.iloc[0,j]
                Year = y
                Rate = f.iloc[i,j]
                rates.append({'LoadPort':LoadPort, 'DischargePort': DischargePort, 'Year':Year,'Rate':Rate})
            
    rate_data = pd.DataFrame(rates)
    
    """Also initialise the temp df with index of total. Temp df is tol hold the dataseries needed to calculate the freight"""
    df = pd.DataFrame(index=total.index)
    df['Date'] = df.index
    
    """This function allows us to apply the expiration date for the wti futures used to determine what structure we apply to the CMA
    Have tried timing and slight improvment with the blow of 0.2seconds...."""
   
    t = time.process_time()

    for_dates = lambda x: (expiry_table.loc[(expiry_table.index.month == x.month)&(expiry_table.index.year == x.year)]['Expiry']).iat[0]
   
    df['Expiry'] = df['Date'].apply(for_dates)
    df.drop(['Date'], inplace=True, axis=1)
    
    
    
    

    print("df['Expiry'] created successfully: Time was {}".format(time.process_time() - t))
    print("Temp DataFrame created successfully")
    print("import_data() created successfully: Time was {}".format(time.process_time() - t2))
    
    return assay, ws, ports, total, rate_data, sub_to_ws, df, basrah_ws_base, crude_diffs, forties_sulphur, exceptions, crudes_to_shift

#crude = 'Amna'
#destination = 'Rotterdam'
#import_data()
#assay, ws, ports, total, rate_data, sub_to_ws, df, basrah_ws_base, crude_diffs = import_data()    
     
def arb(crude,destination,assay, ws, ports, total, rate_data, sub_to_ws, df, basrah_ws_base, crude_diffs, forties_sulphur, exceptions, crudes_to_shift): 
    #crude = 'Azeri'
    #destination = 'Rotterdam'
    
    #crude = 'Amna'
    #destination = 'Houston'
    
    """create the dataframes for use later"""
    df_freight = pd.DataFrame(index=df.index)
    df_prices = pd.DataFrame(index=df.index)
    
    index_wti = [x.lower().strip() for x in ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI','WTI']]
    index_dtd = [x.lower().strip() for x in ['DATED BRENT', 'DATED','N.SEA DATED','BTC Dated', 'MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','    GHANA DATED']]
    index_dub = [x.lower().strip() for x in ['DUBAI','DUBAI M2','OMAN/DUBAI']]
    crudes_diff_against_osp = crudes_to_shift
    
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
    sub_region = ports[ports['Name'] == loadport]['Subregion'].map(sub_to_ws[1]).to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
    sub_region_2 = ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[1]).to_string(index = False)
    discharge_price_region = ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[3]).to_string(index = False)

    expiry_condition = df.index < df.Expiry
    cfd_condition = cfd1 > cfd2

    """Need to handle the one month forward OSP concept, so here, take the dataframe for the exceptions above, condense to monhtly values which wont
    change the value as same for each day, shift that forward then re-expand"""
    if assay[crude]['Code'] == 'multiple':     
        diff = total[exceptions[crude][discharge_price_region]['Code']]
        crude_vs = exceptions[crude][discharge_price_region]['Index'].lower().strip()
    else:    
        diff = total[assay[crude]['Code']]
        crude_vs = assay[crude]['Index'].lower().strip()
        
        
    """This is to make sure we use the CIF port if within the loaidng region, or switch to FOB loading port and FOB price if not"""
    if (assay[crude]['Basis'] == 'CIF') & (sub_region_2 != sub_region):
        loadport = assay[crude]['FOBLoadPort']
        sub_region = ports[ports['Name'] == loadport]['Subregion'].map(sub_to_ws[1]).to_string(index = False)
        diff = total[assay[crude]['FOBCode']]
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
            
            
            
            if ports[ports['Name'] == destination]['Country'].iat[0] == 'South Korea':
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
                df_freight['Buzzard_Content'] = forties_sulphur['buzzard content']
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
                func_ws_base = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['SOMO']).iat[0]
                func_fr = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['FR']).iat[0]
                func_bhapi = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['BHAPI']).iat[0]                
                func_blapi = lambda x: (basrah_ws_base.loc[(basrah_ws_base.index.year == x.year)]['BLAPI']).iat[0]                
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
                #df_freight[['WS for Esc','API Esc']] = df_freight[['WS for Esc','API Esc']].resample('MS').mean().shift(-1, freq='MS').reindex(total.index).fillna(method='ffill')

             # South Korean particulars
            if ports[ports['Name'] == destination]['Country'].iat[0] == 'South Korea':
                # Freight rebate on imported crudes
                df_freight['Murban_Freight_Comp'] = total['PFAOC00'] / 100 * df_freight['Murban_Sing_Flat'] / 7.66 #Murban density conversion
                df_freight['UKC-Yosu_VLCC'] = total['AASLA00'] * 1000000 / 2000000
                df_freight['Freight_Rebate'] = np.maximum(df_freight['UKC-Yosu_VLCC'] - df_freight['Murban_Freight_Comp'], 0.6)
                df_freight['Costs'] -= df_freight['Freight_Rebate']
                
                # Tax rebate on crudes out of Europe
                if ports[ports['Name'] == loadport]['Region'].iat[0] in (['NW EUROPE','MED']):
                    df_freight['FTA_Tax_Rebate'] = 0.006 * total['LCOc1']
                    df_freight['Costs'] -= df_freight['FTA_Tax_Rebate']
                
                # Tax rebate on crudes out of the US
                if ports[ports['Name'] == loadport]['Region'].iat[0] in (['N AMERICA']):
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

            """This finds the correct worldscale rate and adjusts if it is lumpsum"""
            ws_codes = ws[(ws['Origin'] == sub_region)&(ws['Destination'] == sub_region_2)]
            
            vessel_size = []
            for i in list(ws_codes['Code']):
                #i = 'PFAGN10'
                size = ws_codes[ws_codes['Code'] == i]['Size'].iat[0]
                vessel_size.append(size)
                name = ws_codes[ws_codes['Code'] == i]['Name'].iat[0]
                if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
                    df_freight[name] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000
                    df_freight[size] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['bbls'].values * 1000) + df_freight['Costs']
                    df_freight.drop(['Rate'], axis=1)
                else:
                    df_freight[name] = total[i]
                    df_freight[size] = total[i] / 100 * df_freight['Rate'] / assay[crude]['Conversion'] + df_freight['Costs']
            
            if 'WS for Esc' in df_freight.columns.values:
                for i in vessel_size:
                    df_freight[i] = df_freight[i] - df_freight['WS for Esc'] + df_freight['API Esc']
                                
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
        
        index_region = ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[2]).to_string(index = False)
        func_list = {'wti':convert_wti, 'dtd':convert_dtd, 'dub':convert_dub}
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


#temp.resample('W-FRI').mean()



    
   