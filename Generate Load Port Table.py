# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:45:19 2018

@author: mima
"""

import pandas as pd 

data = pd.ExcelFile('C://Users//mima//Documents//FlatRatesComplete.xlsx')
rates = []
lplen = len(f.iloc[:,1])
dplen = len(f.iloc[1,:])
for x,y in enumerate([name.split()[2] for name in data.sheet_names]):
    f  = pd.read_excel(data, sheet_names = x, header = None).iloc[1:46,1:]
    for j in range(1, dplen):
        for i in range(1,lplen):
            LoadPort = f.iloc[i,0]
            DischargePort = f.iloc[0,j]
            Year = y
            Rate = f.iloc[i,j]
            rates.append({'LoadPort':LoadPort, 'DischargePort': DischargePort, 'Year':Year,'Rate':Rate})
            
flat_rate_table = pd.DataFrame(rates)




years = [name.split()[2] for name in data.sheet_names]

f  = pd.read_excel(data, sheet_names = 1, header = None).iloc[1:46,1:]
len(f.columns)
f.tail(15)

lplen = len(f.iloc[:,1])
dplen = len(f.iloc[1,:])



f.iloc[0,0]

f.iloc[1,0]

f.shape
for k in enumerate([name.split()[2] for name in data.sheet_names]):
    print(k)


    

print(i)



for i in range(1,flen+1):
    print(i)