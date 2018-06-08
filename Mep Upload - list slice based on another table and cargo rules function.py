# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:13:37 2018

@author: mima
"""

import pandas as pd
import numpy as np
from datetime import datetime

#TARGO_DATA = pd.read_excel('L://TRADING//ANALYSIS//BALANCES//NWE BALANCE.xlsm', sheetname = 'TARGO Stems')
med_data = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx', sheetname = 'Sheet3')

med_data = pd.read_excel('M:\meddata.xlsx')
targo_vessel_list = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx', sheetname = 'Targo Vessel')
grade_table = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx', sheetname = 'Port names')

####################################################################
#Compare vessel lists to see what we are missing and replace with empty values whilst keeping the vessels themselves
####################################################################

F31_vessel_list = pd.DataFrame(med_data['Vessel'].unique()).rename(columns={0:'Vessel'})
merged_list = pd.merge(F31_vessel_list,targo_vessel_list, how='left',indicator=True)
ToBeReplaced = merged_list[merged_list['_merge'] == 'left_only']
ToBeReplaced = list(ToBeReplaced['Vessel'])
med_data['Vessel'] = med_data['Vessel'].replace(ToBeReplaced,'')


####################################################################
# Replace the Load Locations with corrected ones from Rachits listy
####################################################################


#dfFil = med_data.loc[for x in med_data['Grade'] if  grade_table['Grade'],:]

med_data = med_data[med_data['Grade'].isin(grade_table['Grade'])]
med_data.loc[list(grade_table['Grade']),:]['Port']=grade_table['Port']

for i,j in grade_table.iterrows():
    med_data.loc[med_data.Grade == j.Grade, 'Load Location'] = j.Port

    

####################################################################
# GET NORMAL CARGO SIZES
####################################################################

def cond_cpc(i):
    if i < 500:
        return 330
    elif 500 <= i  < 600:
        return 535
    elif 600 <= i < 720:
        return 670
    elif 720 <= i < 760:
        return 760
    else:
        return i
    

def cond_other(i):
    if i < 300:
        return 250
    elif 300 <= i  < 400:
        return 315
    elif 400 <= i < 500:
        return 450
    elif 500 <= i < 600:
        return 510
    elif 600 <= i < 700:
        return 630
    elif 700 <= i < 900:
        return 750
    elif 900 <= i <= 1200:
        return 1020
    elif i > 1200:
        return 2050
    else:
        return i
    
       
def standardize_cargos(cargo):
    if cargo['Grade'] == 'CPC BLEND':
        return cond_cpc(cargo['Quantity'])
    else:
        return cond_other(cargo['Quantity'])
    
        
med_data['Quantity'] = med_data.apply(standardize_cargos, axis=1)


 ####################################################################
# comapre the two vessel lists and see which ones are missing from targo that are in the file and remove from the larger dataframe
 ###################################################################


# =============================================================================
# look for cargo co load where sum quanitty is greater than expected
# =============================================================================

df = df.sortby('loaddate').reset_index(drop=True)
df = df.loc[(df['dischargedate']>df['loaddate'].shift(-1)) | (df['loaddate']<df['dischargedate'].shift(1))] #give me co loads
df = df.sortby(mike.loaddate)



for row in df.itterows():
    index, data = row
    

#cpc_nov = med_data.loc[(med_data.Grade == 'CPC BLEND') & (med_data['Laycan From'] > datetime(2017,10,31))]

test_path='L:/TRADING/ANALYSIS/Python/MIMA/InputTargoStuff/'
med_data.to_excel(test_path+'CLEANED_MED.xlsx')