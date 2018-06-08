# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:58:37 2018

@author: mima

Need flat rate table
"""
import pandas as pd

data = pd.ExcelFile('C://Users//mima//Downloads//toydata.xlsx')
assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
ws = pd.read_excel(data, 'ws')
ws_table = pd.read_excel(data, 'ws_table', header = 1)
flat_rate = pd.read_excel(data, 'flat_rate')
prices = pd.read_excel(data, 'prices', header = 1)
ports = pd.read_excel(data, 'ports')
products = pd.read_excel(data, 'rott products')




crude = 'Azeri'
discharge = 'Houston'

def fr(crude, discharge):
    
    rate = flat_rate[(flat_rate['LoadPort'] == assay[crude]['LoadPort'])&
                     (flat_rate['DischargePort'] == discharge)&
                     (2018 == flat_rate['Year'])]['Rate'].values
    
    return rate
                  


fr(crude, 'Houston')




ws_table.head()

total = prices.merge(ws_table, how = 'inner', left_index = True, right_index = True)

ports[.loc[discharge]

""" Need to have regions etc sorted - remember to cha nge the codse below"""
   
def world_scale():
    """This finds the correct worldscale rate and adjusts if it is lumpsum"""
    ws_codes = ws[(ws['Origin'] == ports[ports['Name'] == assay[crude]['LoadPort']]['Region']) &
                      (ws['Destination'] == ports[ports['Name'] == discharge]['Subregion'].values)]

    freight_frame = pd.DataFrame()

    for i in list(ws_codes['Code']):
        if ws_codes[ws_codes['Code'] == i]['Terms'].values == 'lumpsum':
            dollars_barrel = prices[ws_codes[ws_codes['Code'] == i]['Code'].values] * 1000000 / (ws_codes[ws_codes['Code'] == i]['Size'].values * 1000)
            freight_frame.append(dollars_barrel)
        else:
            dollars_barrel = prices[i] / 100 * flat_rate / assay[crude]['BT']
            landed_costs.append(dollars_barrel)
            
    return freight_frame


def pricing_adjustment(crude):
    index_wti = ['WTI F1','WTI CMA','1ST LINE WTI','2D LINE WTI','L.A WTI','FORWARD WTI']
    index_dtd = ['DATED	BRENT DATED','N.SEA DATED','BTC DATED	MED DATED','WAF DATED','CANADA DATED','CANADA BRENT DATED','ANGOLA DATED','	GHANA DATED']
    index_dubai = ['DUBAI','DUBAI M2']
    
    def convert_wti(crude):
        if assay[crude]['Index'] in index_wti:
            if prices.index < prices.wti_expiry_date: 
                diff_vs_wti = prices[assay[crude]['Code']] +  (prices[wtim1] -  prices[wtim2]) 
            else:
                diff_vs_wti = prices[assay[crude]['Code']]
                
        elif assay[crude]['Index'] in index_dtd:
            if prices[first_cfd] > prices[second_cfd]: 3
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
                    
        elif assay[crude]['Index'] in index_dtd:
            
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
    


    
def other_costs("This will have the port costs table and any rebates etc that are needed")