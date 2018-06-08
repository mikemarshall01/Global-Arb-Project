# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 14:12:42 2018

@author: mima
"""

import json 
import pandas as pd
import pyodbc
from datetime import datetime
import dateutil.parser

# =============================================================================
# Open the text file, read it and assign to varibale
# pass that variable through the pd.reader as format must be familiar
# =============================================================================

json_data =  open('L:\\TRADING\SQLDataUpload\PETTest.txt','r').read()
df = pd.read_json(json_data)
df['Date'] = [row[0] for row in df['data']]
df['Amount'] = [row[1] for row in df['data']]


# =============================================================================
# Below is not neededed really, but good example of list comprehension
# =============================================================================
df['Newdate'] = [row+"01" if len(row) <8 else row for row in df['Date']]


# =============================================================================
# first we need to take the string and turn that into a datime object and then 
# create a string from the datetime in our format. Below are examples of using 
# strptime, stfrtime and dateutil.parse which turns into a datetime object
# =============================================================================
df['mike'] = [datetime.strptime(row,'%Y%m%d').strftime('%d-%m-%Y') 
                                    for row in df['Newdate']]
df['last_updated2'] = [dateutil.parser.parse(row).strftime('%d-%m-%Y') 
                                    for row in df['last_updated']]


# =============================================================================
# Checking for column headers here
# =============================================================================
df.columns


# =============================================================================
# This is how you connect python to a databse
# =============================================================================
connection = pyodbc.connect('''Driver={SQL Server}; 
                            Server=STUKLS022; 
                            Database=TradeTracker; 
                            uid=GAMI ;
                            pwd=pw;
                            Trusted_Connection=yes''')    
cursor = connection.cursor()
SQLCommand = ('''INSERT INTO EiaSeriesTest"
              "(Id, Name, CategoryID, f, units, updated)"
              "VALUES (?,?,1,?,?,?)''')

Values = list(zip(df.series_id, df.name, df.f, df.units, df.last_updated2)) 

try:
    cursor.executemany(SQLCommand,Values)
except:
    print(Values)
    raise
        
connection.commit()
connection.close()




