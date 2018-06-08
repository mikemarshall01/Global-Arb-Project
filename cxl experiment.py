# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:54:01 2017

@author: mima
"""
import pandas as pd

file = pd.read_excel('L:/TRADING/ANALYSIS/PRICES/CXL Prices.xlsx', sheetname=0)
print(file.head())

unique_name_list = pd.Series(file['name'].unique())

from datetime import datetime

filter = file[(file['name'] == 'P Brent BFOE JUN FWD') &
              (file['asof_date'] > datetime(2017,4,1))]