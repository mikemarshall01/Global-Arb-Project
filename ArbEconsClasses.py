# -*- coding: utf-8 -*-
"""
Created on Sat Mar 3 18:09:50 2018

@author: mima

Need flat rate table
"""

#import_data()

import pandas as pd
import numpy as np
from datetime import datetime as dt
import random
import time
    


class import_data:
    

    #crude = 'Forties'
    #destination = 'Houston'
    def __init__ (self, assay, ws, ports, total, rate_data, sub_to_ws, df):
        #self.get_data = assay, ws, ports, total, rate_data, sub_to_ws, df
        self.assay = assay
        self.ws = ws
        self.ports = ports
        self.total = total
        self.rate_data = rate_data
        self.sub_to_ws = sub_to_ws
        self.df = df 

    @staticmethod  
    def get_data():
        t2 = time.process_time()
        #data = pd.ExcelFile('C://Users//mike_//Downloads//toydata1004 (1).xlsx')
        #raw_rates = pd.ExcelFile('C://Users//mike_//Downloads//FlatRatesComplete (1).xlsx')
        
        data = pd.ExcelFile('C://Users//mima//Documents//toydata1004.xlsx')
        raw_rates = pd.ExcelFile('C://Users//mima//Documents//FlatRatesComplete.xlsx')
        
        assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
        ws = pd.read_excel(data, 'ws')
        ws_table = pd.read_excel(data, 'ws_table', header = 1)
        #rate_data = pd.read_excel(data, 'flat_rate')
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
        This turns the 5 years of rate matricies into a table for use to reference - 12/04/2018
        """    
        
        
        rates = []
        
        for x,y in enumerate([name.split()[2] for name in raw_rates.sheet_names]):
            f  = pd.read_excel(raw_rates, sheetname = x, header = None).iloc[1:46,1:]
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
        
        """
        Also initialise the temp df with index of total.
        Temp df is tol hold the dataseries needed to calculate the freight
        """
        df = pd.DataFrame(index=total.index)
        
        """
        This function allows us to apply the expiration date for the wti futures used to determine what structure we apply to the CMA
    
        Have tried timing and slight improvment with the blow of 0.2seconds....
        """
       
        t = time.process_time()
  
        def apply_expiry(x):
            return (expiry_table.loc[(expiry_table['Month'].dt.month == x[1])&
                                          (expiry_table['Month'].dt.year == x[0]), 'Expiry']).iat[0]
                
        #v_apply_expiry =  np.vectorize(apply_expiry)
        index_for_func = np.array((list(df.index.year),list(df.index.month))).transpose()
        df['Expiry'] = np.apply_along_axis(apply_expiry,1,index_for_func)
        print("df['Expiry'] created successfully: Time was {}".format(time.process_time() - t))
        print("Temp DataFrame created successfully")
        print("import_data() created successfully: Time was {}".format(time.process_time() - t2))
        
        return assay, ws, ports, total, rate_data, sub_to_ws, df
    
    assay, ws, ports, total, rate_data, sub_to_ws, df = get_data.__func__()
    
    
def arb(crude,destination,):
    assay = import_data.assay
    ws = import_data.ws
    ports = import_data.ports
    total = import_data.total
    rate_data = import_data.rate_data
    sub_to_ws = import_data.sub_to_ws
    df = import_data.df
    
    
    """
    create the flat rates table for the rates calculations and column creation
    """
    
    def calculate_flat_rate():
        
        flat_rate_table = rate_data.loc[(rate_data['LoadPort'] == assay[crude]['LoadPort'])&
                  (rate_data['DischargePort'] == destination)]
       
        return flat_rate_table
    
    flat_rate_table = calculate_flat_rate()
    print("flat_rate_table created successfully")


    """
    THIS IS ESSENTIALLY A VLOOKUP
    
    inputs are the table to look up values from for each row of another table
    
    in this case we use it to look up the year in the dataframes row and take the year and match it against the corresponding row (or rows? possibly but need to investigate)
    and return the value in the rate row.
    
    use the .iat function to return the inger values - this speeds up operations
    
    x refers to the row in the dataframe we are applying to and the name is the index reference of the row - which is the dateindex.
    Hence, since you cant take the index.year directly, must turn the date into a string, get the date as a dt object from the string and then get the year
    
    """
    
    #x = '2017-12-06 00:00:00'
    #type(dt.strptime(str(x), '%Y-%m-%d %H:%M:%S').year)
    #flat_rate_table = 
    def calculate_flat_rates(x):
        return float(flat_rate_table.loc[flat_rate_table['Year'].astype(int) == x, 'Rate'])
    
    """ 
    Vectorising the function amkes it applicable over an array - before had to use pandas which was element wise application - i.e. SLOW
    """
    
    v_calculate_flat_rates = np.vectorize(calculate_flat_rates)
    df['Rate'] = np.apply_along_axis(v_calculate_flat_rates,0,np.array(df.index.year))
    
    #df['Rate'] = df.index.to_series().apply(lambda x: calculate_flat_rates(x))
    
    print("df['Rate'] created successfully")
    
    
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
        
        sub_region_2 = ports[ports['Name'] == destination]['Subregion']
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
    print("calculate_world_scale() created successfully")
    
    #def pricing_adjustment():
    index_wti = [x.lower().strip() for x in ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI']]
    index_dtd = [x.lower().strip() for x in ['DATED BREN, DATED','N.SEA DATED','BTC Dated', 'MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','    GHANA DATED']]
    index_dub = [x.lower().strip() for x in ['DUBAI','DUBAI M2']]
    
    
    
    
    
    indicies = {'index_wti': index_wti, 'index_dtd':index_dtd,'index_dub':index_dub}
    #TEST PRICES
    def generate_price(row, minp, maxp):
        return row + random.uniform(minp,maxp)

    # will need something to determine which cfds to choose - for now thw ones to choose are going to be the 3/4
    
   
    dtd = total['PCAAS00']
    wtim1 = total['CLc1'] 
    wtim2 = total['CLc2']
    cfd1 = total['PCAKG00']
    cfd2 = total['AAGLV00']
    efpm2 = total['AAGVX00']
    brentm2 = total['LCOc2']
    efs2 = total['AAEBS00']
    crude_vs = assay[crude]['Index'].lower().strip()
    diff = total[assay[crude]['Code']]
    
    expiry_condition = df.index < df.Expiry
    cfd_condition = cfd1 > cfd2
    
    def convert_wti():
        if crude_vs in index_wti:
            df['diff_vs_wti'] = np.where(expiry_condition,
                                          diff + (wtim1 -  wtim2),
                                          diff)
        elif crude_vs in index_dtd:
            df['diff_vs_wti']  = np.where(cfd_condition,
                                          diff + cfd2 +  efpm2 + (brentm2 - wtim2),
                                          diff + cfd1 +  efpm2 + (brentm2 - wtim2))
        elif crude_vs in index_dub:
            pass
        else:
            df['diff_vs_wti'] = diff
        return df
        
    convert_wti()
    print("convert_wti() created successfully")
    
    def convert_dtd():
        conditions = [(expiry_condition & cfd_condition),
                      (expiry_condition & np.invert(cfd_condition)),
                      (np.invert(expiry_condition) & cfd_condition),
                      (np.invert(expiry_condition) & np.invert(cfd_condition))]
    
        choices = [(diff +  (wtim1 -  wtim2) - (brentm2 - wtim2) - efpm2 - cfd2),
                   (diff +  (wtim1 -  wtim2) - (brentm2 - wtim2) - efpm2 - cfd1),
                   (diff - (brentm2 - wtim2) - efpm2 - cfd2),
                   (diff - (brentm2 - wtim2) - efpm2 - cfd1)]

        if crude_vs in index_wti:
            df['diff_vs_dtd'] = np.select(conditions, choices)
            
        elif crude_vs in index_dtd:
            df['diff_vs_dtd'] = np.where(cfd_condition,
                                         diff + (4 * (cfd2 - cfd1) / 14),
                                         diff + (4 * (cfd1 - cfd2) / 14)) # the 4 will be replaces by the journey time between thge two ports

        elif crude_vs in index_dub:
            pass
        
        else:
            df['diff_vs_dtd'] = diff
            
        return df['diff_vs_dtd']
    
    convert_dtd()
    print("convert_dtd() created successfully")
    
    def convert_dub():
        if crude_vs in index_wti:
            df['diff_vs_dub'] = np.where(expiry_condition,
                                          diff + (wtim1 -  wtim2)  - (brentm2-wtim2) + efs2,
                                          diff - (brentm2-wtim2) + efs2)
              
        elif crude_vs in index_dtd:
            df['diff_vs_dub'] = np.where(cfd_condition, 
                                          diff + cfd2 +  efpm2 + efs2, 
                                          diff + cfd1 +  efpm2 + efs2)
        
        elif crude_vs in index_dub:
            pass
        
        return df['diff_vs_dub'] 
    
    convert_dub()
    print("convert_dub() created successfully")
    print(df.head())
   
    return df
    
    #t = time.process_time()
    #pricing_adjustment()
    #print("pricing_adjustment() created successfully: Time was {}".format(time.process_time() - t))
    
    #return df



#arb(crude, destination)

#def other_costs("This will have the port costs table and any rebates etc that are needed")



#arb(crude, destination)

#def other_costs("This will have the port costs table and any rebates etc that are needed")