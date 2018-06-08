# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:13:37 2018

@author: mima
"""

import pandas as pd
import numpy as np

TARGO_DATA = pd.read_excel('L://TRADING//ANALYSIS//BALANCES//NWE BALANCE.xlsm', sheetname = 'TARGO Stems')
med_data = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx')
targo_vessel_list = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx', sheetname = 'Targo Vessel')
grade_table = pd.read_excel('L:\TRADING\ANALYSIS\Python\MIMA\InputTargoStuff\TARGO Upload History Med with Validation V2.xlsx', sheetname = 'Port names')
grade_list = list(med_data['Grade'].unique())
print(grade_list)

####################################################################
# GET PORT LOAD LOCATIONS FROM WHAT WE HAVE ALREAdY
####################################################################

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

F31_vessel_list = pd.DataFrame(med_data['Vessel'].unique()).rename(columns={0:'Vessel'})
merged_list = pd.merge(F31_vessel_list,targo_vessel_list, how='left',indicator=True)
ToBeReplaced = merged_list[merged_list['_merge'] == 'left_only']
ToBeReplaced = list(ToBeReplaced['Vessel'])
med_data['Vessel'] = med_data['Vessel'].replace(ToBeReplaced,'')

for i,j in grade_table.iterrows():
    med_data.loc[med_data.Grade == j.Grade, 'Load Location'] = j.Port




med_data.to_excel('M:\med_corrected.xlsx')








quantities = pd.concat([med_data['Cargo ID'],med_data['Grade'],med_data['converted'], med_data['Vessel'], med_data['Equity']], axis = 1)

med_data.loc[med_data['Vessel']ToBeReplaced]

df.replace(med_data[med_data['Vessel']])



conditions_other = [(lambda i: i < 300, 250),
                    (lambda i: 300 <= i  < 400, 315),
                    (lambda i: 400 <= i < 500, 450),
                    (lambda i: 500 <= i < 600, 510),
                    (lambda i: 600 <= i < 700, 630),
                    (lambda i: 700 <= i < 900, 750),
                    (lambda i: 900 <= i <= 1200, 1020),
                    (lambda i: i > 1200, 2050)]

conditions_other[0]
                    
conditions_cpc = [(lambda i: i < 500, 330),
                  (lambda i: 500 <= i < 600, 535),
                  (lambda i: 600 <= i < 720, 670),
                  (lambda i: 720 <= i <= 760, 760)]


   
cond_other(100)

def choose_condition(k):
    if i == 'CPC Blend':
        conditions_cpc(250)
    else:
        conditions_other
        
x = choose_condition('CPC Blend')

new_df = pd.concat([med_data['Quantity'], med_data['Grade']])

def apply_condition(df):
    if df['Grade'] == 'CPC Blend':
        for condition, replacement in conditions_cpc:
            if condition(df['Quantity']):
                return df['replacement']
    else:
        for condition, replacement in conditions_other:
            if condition(df['Quantity']):
                return df['replacement']
    return df['Quantity']
apply_condition(med_data)
def apply_condition(quantity, grade):
    if grade == 'CPC Blend':
        for condition, replacement in conditions_cpc:
            if condition(quantity):
                return replacement
    else:
        for condition, replacement in conditions_other:
            if condition(quantity):
                return replacement
            
for row in med_data:
    apply_condition(med_data['Quantity'], med_data['Grade'])
            
            
            
med_data['Converted'] = med_data['Quantity'].apply()


return replacement if condition(i) else i for condition, replacement in conditions_cpc




[x(2520) for x in [conditions_cpc if row == 'CPC BLEND' else conditions_other for row in med_data['Grade']]]

[replacement for function, replacement in conditions_cpc if row == 'CPC BLEND' else conditions_other for row in med_data['Grade'] if function(i)]


def standardise_cargo_size(i):
     [replacement for function, replacement in 
            [conditions_cpc if row == 'CPC BLEND' else conditions_other(250) for row in med_data['Grade']]]
    
    

def standardise_cargo_size_attempt(i):
     [replacement for condition, replacement in 
            [conditions_cpc if row == 'CPC BLEND' else conditions_other for row in med_data['Grade']] if condition(250)]
    
standardise_cargo_size(200) 
    
def standardise_cargo_size2(i):
    for condition, replacement in conditions_other:
        if condition(i):
            return replacement
    return i
#
def standardise_cargo_size3(i,j):
    [replacement for condition, replacement in conditions_other 
     if condition(i)]
    
standardise_cargo_size(med_data['Quantity'])

ar = list(map(standardise_cargo_size2, med_data['Quantity']))

ar = pd.Series(map(standardise_cargo_size, med_data['Quantity']))

print(ar)


print(ar)    




    

### This si the column we want to change 
med_data['Quantity_Adjusted'] = 


