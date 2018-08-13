# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:52:53 2018

@author: mima
"""

import pandas as pd
from pandas.tseries.offsets import MonthEnd
import numpy as np


data = "C://Users//mima//Desktop//EXPORTS_UPLOAD_SHEET.xlsx"
sheet = pd.read_excel(data).set_index(['Grade','Destination Location'])
test = pd.read_excel(data)
test_var_values = test.columns.drop(['Grade','Destination Location'])
test_melt = pd.DataFrame(pd.melt(test, id_vars = ['Grade','Destination Location'], value_vars = test_var_values)).rename(columns={'variable':'Laycan To', 'value':'Quantity'})
test_melt['Laycan From'] =  test_melt['Laycan To'] + MonthEnd(1)
test_melt['Unit'] = 'kb'
test_melt['Balance Item'] = 1
conditions = [
        (test_melt['Grade']=='CANADA CRUDE SWEET'),
        (test_melt['Grade']=='CANADA CRUDE SOUR'),
        (test_melt['Grade']=='US CRUDE SWEET'),
        (test_melt['Grade']=='US CRUDE SOUR'),
        (test_melt['Grade']=='S AMERICA CRUDE SWEET'),
        (test_melt['Grade']=='S AMERICA CRUDE SOUR'),
        (test_melt['Grade']=='CARIBBEAN CRUDE SWEET'),
        (test_melt['Grade']=='CARIBBEAN CRUDE SOUR')]
choices = ['CANADA EC','CANADA EC','US GULF (padd 3)','US GULF (padd 3)','S AMERICA','S AMERICA','CARIBBEAN','CARIBBEAN']
test_melt['Load Location'] = np.select(conditions, choices, 'NULL')
test_melt = test_melt[['Grade','Quantity','Unit','Laycan To','Laycan From','Load Location','Balance Item','Destination Location']]
writer = pd.ExcelWriter('L://TRADING//ANALYSIS//Python//MIMA//TARGO_UPLOAD_FEED.xlsx')
test_melt.to_excel(writer,'FEED')
writer.save()
