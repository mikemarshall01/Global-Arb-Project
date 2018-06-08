# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta 

ref_data = pd.read_excel("L://TRADING//ANALYSIS//REFINERY DATABASE//Latest ref sheets//Latest//EVERYTHING.xlsx",
                        sheetname=0, header=0)
test_path='L:/TRADING/ANALYSIS/Python/MIMA/RefineryDatabaseChecks/'
#ref_data = pd.read_excel("M://test_data_for_ref_resample.xlsx", sheetname=0, header=0)
#ref_data['Start_Date'].apply(lambda x : x.date() in ref_data['Start_Date'])


# =============================================================================
### slice maintenance list to include only the crude and the condensate units that we are interested in
# =============================================================================

ref_data = ref_data[(ref_data['UNIT_NAME'] == 'CRUDE #1')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #2')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #3')|
                     (ref_data['UNIT_NAME'] == 'CRUDE #4')|
                     (ref_data['UNIT_NAME'] == 'CONDENSATE DISTILLATION #1')|
                     (ref_data['UNIT_NAME'] == 'CONDENSATE DISTILLATION #2')]

#ref_data = ref_data[(ref_data['UNIT_NAME'] == 'CRUDE #1')]

### Set start and end dates for the entire dataframe
pd_index_start_date = datetime(1900, 1, 1)
pd_index_end_date = datetime(2030, 12, 1)


### define the length of the dataframe 
#total_time_period = pd_index_end_date - pd_index_start_date


### create the index by creating a list of days from the start date, 
### in days, for the duration of the time period detrmined above and 
### add one due to zero indexing
#pd_index_days_list=[pd_index_start_date + timedelta(days=x) 
#        for x in range(total_time_period.days +1)]

###ORRRRRR.......

# =============================================================================
### create a list of dates, daily, between the start and end dates
# =============================================================================

pd_index_days_list = pd.date_range(pd_index_start_date, pd_index_end_date, 
                                   freq = 'D')


# =============================================================================
### create a list of all individual refineries for column headers
# =============================================================================

ref_list=pd.Series(ref_data['REFINERY'].unique())



# =============================================================================
### create the dataframe with the col headers and index determined above
# =============================================================================

maintenance_framework = pd.DataFrame(0, index= pd_index_days_list, columns=ref_list)


# =============================================================================
### for each row in the database (each row represents a maintenance outage)
### take the start and end dates to get duration from cols 4 and 5
### get the name from column 1, then create a date range for that maintenance 
### 
# =============================================================================

for index, row in ref_data.iterrows():
    end_date = row[5]
    #print(row)
    start_date = row[4]
    name = row[1]
    #num_days_offline = end_date - start_date
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    ref_test_df = pd.DataFrame(row['CAP_OFFLINE'], index=date_range, columns= [row['REFINERY']])
    maintenance_framework.loc[ref_test_df.index,name]=maintenance_framework.loc[ref_test_df.index,name]+ref_test_df.loc[ref_test_df.index,name]
    #ref_test_df[name] = row['CAP_OFFLINE']
    #ref_sum_df = ref_test_df.sum(axis=1).to_frame(name)
    
    #maintenance_framework = maintenance_framework.append(ref_test_df)
    
    #maintenance_framework = pd.merge(maintenance_framework, ref_test_df,
    #                                 how='left', left_index = True, right_index = True)
    
    #maintenance_framework = maintenance_framework.join(ref_test_df, how = 'outer')
    
    #maintenance_framework.groupby(by=name,axis=1).sum()

### create dataframe for refinery maintenance line
### first take duration of maintenance

#num_days_offline = end_date - start_date

### then create time series index for duration period
#ref_test_index=[start_date + timedelta(days=x) 
#        for x in range(num_days_offline.days +1)]

### then create dataframe for this index
#ref_test_dataframe  = pd.DataFrame(data=None, index=ref_test_index)
#ref_test_dataframe['Refinery'] = ref_data.loc[0,'CAP_OFFLINE'] 




#ref_data['Duration']
#
#
#pd_index_days_list=[pd_index_start_date + timedelta(days=x) 
#        for x in range(total_time_period.days +1)]
#
#
#
#
#
#
##start_date = ref_data.loc[0,4]
##end_date = ref_data.loc[0,5]
#cap_offline = ref_data.loc[0,6]
#
#num_days_offline = end_date - start_date
#
#days_list=[x for x in range(start_date,end_date)]
#dfDiff=pd.DataFrame(0,index=days_list)
#
#
#ref_data['Diff']=ref_data['END_DATE']-ref_data['Start_Date']
#dfDiff.loc[ref_data['Start_Date']]=ref_data.loc[ref_data['Start_Date'],'Diff']
#
#total_time_period = pd_index_end_date - pd_index_start_date
#
#maintenance_framework = pd.DataFrame(data=None, index= pd_index_days_list)
    
filtered_start_date = date(2005,1,1)
filtered_end_date = date(2020,12,1)    
d = maintenance_framework.index.day - np.clip((maintenance_framework.index.day-1) // 10, 0, 2)*10 - 1
date = maintenance_framework.index.values - np.array(d, dtype="timedelta64[D]")
maintenance_framework_decade = maintenance_framework.groupby(date).mean()  

maintenance_framework_month = maintenance_framework.resample('M', convention = 'start').mean()
maintenance_framework_week =  maintenance_framework.resample('W-FRI', convention = 'start').mean()

maintenance_framework_month = maintenance_framework_month.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework_week = maintenance_framework_week.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework_decade = maintenance_framework_decade.loc[filtered_start_date:filtered_end_date,:]
maintenance_framework = maintenance_framework.loc[filtered_start_date:filtered_end_date,:]

maintenance_framework.to_excel(test_path+'daily.xlsx')
maintenance_framework_month.to_excel(test_path+'monthly.xlsx')
maintenance_framework_week.to_excel(test_path+'weekly_2.xlsx')
maintenance_framework_decade.to_excel(test_path+'decade.xlsx')

    
