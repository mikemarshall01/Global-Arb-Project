# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:07:04 2017

@author: mima
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta 

ref_data = pd.read_excel("L://TRADING//ANALYSIS//REFINERY DATABASE//Latest ref sheets//Latest//REST of WORLD.xlsm",
                        sheetname=0, header=0)

ref_data = ref_data[(ref_data['COUNTRY'] == 'RUSSIAN FEDERATION') & (ref_data['START_DATE'] > datetime(2017,1,1)) & ((ref_data['UNIT_NAME'] == 'CRUDE #1')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #2')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #3')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #4')|
                     (ref_data['UNIT_NAME'] == 'CONDENSATE DISTILLATION #1')|
                     (ref_data['UNIT_NAME'] == 'CONDENSATE DISTILLATION #2'))]
                      
pd_index_start_date = datetime(1900, 1, 1)
pd_index_end_date = datetime(2030, 12, 1)

pd_index_days_list = pd.date_range(pd_index_start_date, pd_index_end_date, 
                                   freq = 'D')

ref_list=pd.Series(ref_data['REFINERY'].unique())

maintenance_framework = pd.DataFrame(0, index= pd_index_days_list, columns=ref_list)

for index, row in ref_data.iterrows():
    end_date = row[5]
    start_date = row[4]
    name = row[1]
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    ref_test_df = pd.DataFrame(row['CAP_OFFLINE'], index=date_range, columns= [row['REFINERY']])
    maintenance_framework.loc[ref_test_df.index,name]=maintenance_framework.loc[ref_test_df.index,name]+ref_test_df.loc[ref_test_df.index,name]
    
    
filtered_start_date = datetime(2005,1,1)
filtered_end_date = datetime(2020,12,1)    
d = maintenance_framework.index.day - np.clip((maintenance_framework.index.day-1) // 10, 0, 2)*10 - 1
date = maintenance_framework.index.values - np.array(d, dtype="timedelta64[D]")
maintenance_framework_decade = maintenance_framework.groupby(date).mean()  

maintenance_framework_month = maintenance_framework.resample('M', convention = 'start').mean()
maintenance_framework_week =  maintenance_framework.resample('W', convention = 'start').mean()

maintenance_framework_month = maintenance_framework_month.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework_week = maintenance_framework_week.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework_decade = maintenance_framework_decade.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework = maintenance_framework.loc[filtered_start_date:filtered_end_date,:]

test_path='L:/TRADING/ANALYSIS/Python/MIMA/RefineryDatabaseChecks/'

maintenance_framework.to_excel(test_path+'daily.xlsx')
maintenance_framework_month.to_excel(test_path+'monthly.xlsx')
maintenance_framework_week.to_excel(test_path+'weekly.xlsx')
maintenance_framework_decade.to_excel(test_path+'decade.xlsx')
