# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:09:30 2017

@author: mima
"""

import pandas as pd
import numpy as np

begin = pd.datetime(2013,1,1)
end = pd.datetime(2013,2,20)

dtrange = pd.date_range(begin, end)

p1 = np.random.rand(len(dtrange)) + 5
p2 = np.random.rand(len(dtrange)) + 10

df = pd.DataFrame({'p1': p1, 'p2': p2}, index=dtrange)

df.dtypes


### calculate decade
d = df.index.day -1 - np.clip((df.index.day-1) // 10, 0, 2)*10
date = df.index.values - np.array(d, dtype="timedelta64[D]")
df.groupby(date).mean()

print(d)

###  What lines 23-25 mean
### we created an index od dtranges bages betweendays as values
print(df.index.day)

### zero-indexing adjustment
print(df.index.day-1)

### divisor and rounded down to nearest whole for 1st, 2nd and 3rd
print((df.index.day-1) // 10)

### numpy's clip takes our array and clips to whole numbers betwen 0 and 2 for 1st, 2bd, 3rd
print(np.clip((df.index.day-1) // 10, 0, 2))

### muliply by 10 for the days
print((np.clip((df.index.day-1) // 10, 0, 2))*10)

### zero indexing for index days as values
print(df.index.day -1 -(np.clip((df.index.day-1) // 10, 0, 2))*10)

### then we take the values of the days in datetime format
print(np.array(d, dtype="timedelta64[D]"))

### vs the values we have created
print(df.index.values)

### put it all together and we get our decades regardless of number of days in month etc
print(df.index.values - np.array(d, dtype="timedelta64[D]"))

### we then group the value in our dataframe by the decade (date variable - notice then 1,1,1,1,11,11,11,11,21,21,21,21)
### and average for your decades
df.groupby(date).mean()