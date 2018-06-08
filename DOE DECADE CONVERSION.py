# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 12:11:55 2018

@author: mima
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta 

DOEAPIDATA = pd.read_excel("L://TRADING//ANALYSIS//BALANCES//BALANCES IN PROGRESS//US Balance.xlsm",
                        sheetname="DOE_API_W", header =0).drop([0])

DOEAPIDATA = DOEAPIDATA.set_index('Date')
DOEAPIDATA = DOEAPIDATA.drop(DOEAPIDATA.columns[0], axis=1)
DOEAPIDATA.index = pd.to_datetime(DOEAPIDATA.index)
DOEAPIDATA = pd.DataFrame(DOEAPIDATA).astype(float)
mike = DOEAPIDATA.resample('d').bfill()
d = mike.index.day -1 - np.clip((mike.index.day-1) // 10, 0, 2)*10
date1 = mike.index.values - np.array(d, dtype="timedelta64[D]")
decade_mike = mike.groupby(date1).mean()
writer = pd.ExcelWriter("M://DOE TEST SHEET.xlsx")
decade_mike.to_excel(writer, 'DOE_API_Dec')
writer.save()

