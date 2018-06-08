# -*- coding: utf-8 -*-
"""
Created on Fri May 11 14:09:40 2018

@author: mima
"""

import pandas as pd

data = pd.ExcelFile('C://Users//mima//Documents//crude list.xlsx')
crude_list = pd.read_excel(data, 'Crudes')


crudes_no_ma = crude_list.loc[~crude_list['Name (or Code)'].str.contains('Monthly Average')]
crudes_only_ma = crude_list.loc[crude_list['Name (or Code)'].str.contains('Monthly Average')]

writer = pd.ExcelWriter('C://Users//mima//Documents//crude codes.xlsx')
crudes_no_ma.to_excel(writer, sheet_name='No Monthly Avg')
crudes_only_ma.to_excel(writer, sheet_name='Only Monthly Avg')
writer.save()