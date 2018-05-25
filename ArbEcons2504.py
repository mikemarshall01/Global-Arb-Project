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


    """Clean the column hedaers so no white spcaes - use simple list comprehension and set headers equal to cleaned"""
    cleaned_column_headers = [i.strip() for i in total.columns.values]
    total.columns = cleaned_column_headers
    
    """The below was get rid of the row in the index that hax NaT against it and then expand to daily and fill backwards"""
    crude_diffs = pd.read_excel(trader_assessed, 'Crude Diffs Traders', header = 0)
    crude_diffs = crude_diffs.loc[pd.notnull(crude_diffs.index)]
    crude_diffs = crude_diffs.resample('D').interpolate().fillna(method='bfill')
    
    """Give me the columns with codes against them"""
    crude_diffs = crude_diffs.drop([name for name in crude_diffs.columns if 'Unnamed' in name], axis=1)
    
    """Slice the crude diffs where the dates in the index are the same as the dates in the total dataframe"""
    crude_diffs = crude_diffs[crude_diffs.index.isin(total.index)]
    
    """Apply the values in crude diffs to the correct codes and dates in the total dataframe"""
    total.loc[total.index.isin(crude_diffs.index), list(crude_diffs.columns)] = crude_diffs[list(crude_diffs.columns)]
    
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
    
    return assay, ws, ports, total, rate_data, sub_to_ws, df


#import_data()
#assay, ws, ports, total, rate_data, sub_to_ws, df = import_data()    
     
def arb(crude,destination,assay, ws, ports, total, rate_data, sub_to_ws, df): 
    #crude = 'Azeri'
    #destination = 'Rotterdam'
    
    #crude = 'Azeri'
    #destination = 'Rotterdam'
    
    """create the dataframes for use later"""
    df_freight = pd.DataFrame(index=df.index)
    df_prices = pd.DataFrame(index=df.index)
    
    index_wti = [x.lower().strip() for x in ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI','WTI']]
    index_dtd = [x.lower().strip() for x in ['DATED BRENT', 'DATED','N.SEA DATED','BTC Dated', 'MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','    GHANA DATED']]
    index_dub = [x.lower().strip() for x in ['DUBAI','DUBAI M2','OMAN/DUBAI']]
    
    """Declare the main prices that will be used in order to use shorthand notation"""
    
    
    dtd = total['PCAAS00']
    dub = total['AAVMR00']
    wtim1 = total['CLc1'] 
    wtim2 = total['CLc2']
    wti_cma_m1 = total['AAVSN00']
    cfd1 = total['PCAKG00']
    cfd2 = total['AAGLV00']
    cfd3 = total['PCAKE00']
    cfd4 = total['PCAKG00']
    cfd5 = total['AAGLU00']
    cfd6 = total['AAGLV00']
    cfd7 = total['AALCZ00']
    cfd8 = total['AALDA00']
    wti_br_m1 = total['WTCLc1-LCOc1']
    wtim1_m2 = total['CLc1-CLc2']
    brentm1_m2 = total['LCOc1-LCOc2']
    efpm2 = total['AAGVX00']
    brentm2 = total['LCOc2']
    efs2 = total['AAEBS00']
    mars_wti2 = total['AAKTH00']
    dfl_m1 = total['AAEAA00']

    days = 5
    sub_region = ports[ports['Name'] == assay[crude]['LoadPort']]['Subregion'].map(sub_to_ws[1]).to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
    sub_region_2 = ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[1]).to_string(index = False)
    discharge_price_region = ports[ports['Name'] == destination]['Subregion'].map(sub_to_ws[3]).to_string(index = False)
    
    expiry_condition = df.index < df.Expiry
    cfd_condition = cfd1 > cfd2

    
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
       
    if assay[crude]['Code'] == 'multiple':     
        diff = total[exceptions[crude][discharge_price_region]['Code']]
        crude_vs = exceptions[crude][discharge_price_region]['Index'].lower().strip()
    else:    
        diff = total[assay[crude]['Code']]
        crude_vs = assay[crude]['Index'].lower().strip()

    
    def construct_freight():
        def calculate_flat_rate():
            """create the flat rates table for the rates calculations and column creation"""    
            flat_rate_table = rate_data.loc[(rate_data['LoadPort'] == assay[crude]['LoadPort'])&
                      (rate_data['DischargePort'] == destination)]
            
            def calculate_flat_rates(x):
                return float(flat_rate_table.loc[flat_rate_table['Year'].astype(int) == x, 'Rate'])
            
            """Vectorising the function amkes it applicable over an array - before had to use pandas which was element wise application - i.e. SLOW"""
            v_calculate_flat_rates = np.vectorize(calculate_flat_rates)
            df_freight['Rate'] = np.apply_along_axis(v_calculate_flat_rates,0,np.array(df.index.year))
            return df_freight
            
        def calculate_freight():

            """This finds the correct worldscale rate and adjusts if it is lumpsum"""
            sub_region = ports[ports['Name'] == assay[crude]['LoadPort']]['Subregion']
            sub_region = sub_region.map(sub_to_ws[1]).to_string(index = False) # NB the index = False is make sure we dont take in the index number given in the output
            sub_region_2 = ports[ports['Name'] == destination]['Subregion']
            sub_region_2 = sub_region_2.map(sub_to_ws[1]).to_string(index = False)
            ws_codes = ws[(ws['Origin'] == sub_region)&(ws['Destination'] == sub_region_2)]
        
            for i in list(ws_codes['Code']):
                size = ws_codes[ws_codes['Code'] == i]['Size'].iat[0]
                name = ws_codes[ws_codes['Code'] == i]['Name'].iat[0]
                if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
                    df_freight[name] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000
                    df_freight[size] = total[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['Size'].values * 1000)
                    df_freight.drop(['Rate'], axis=1)
                else:
                    df_freight[name] = total[i]
                    df_freight[size] = total[i] / 100 * df_freight['Rate'] / assay[crude]['Conversion']                    
            return df_freight
        
        calculate_flat_rate()
        calculate_freight()
        return df_freight

    def convert_prices():
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
                df_prices['cfd3'] = cfd3
                df_prices['cfd5'] = cfd5
                if sub_region == sub_region_2:
                    df_prices['vs_dtd'] = diff
                elif sub_region == 'WAF':
                    df_prices['vs_dtd'] = diff + ((cfd3 - cfd5)/14) * days
                else:
                    df_prices['vs_dtd'] = diff + ((cfd3 - cfd5)/14) * days
 
            #elif crude_vs in index_dub:
                #""" This is because all the eastern crudes heading here have diffs against BWAVE"""
                #pass
                #df_prices['diff'] = diff
                #if assay[crude]['LoadPort'] in ['Basrah']:
                    #df_prices['osp_vs_dtd']
                    #df_prices['freight esc/desc']
                    #df_prices['api esc/desc']
                   # df_prices['osp_vs_dtd']

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
            try:
                """test to see if nan's present
                print('this is the price index')
                print(price_index)
                print(df_prices)
                print(df_prices[price_index].isnull().sum().sum())
                print('this is the freight')
                print(k)
                print(df_freight[k].isnull().sum().sum())"""
                temp[name] = df_prices[price_index].add(df_freight[k])
            except Exception as e: print('temp fails')    
    except Exception as e: print('check')

    return temp
    



    
   