# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 16:07:16 2018

@author: gami
"""
from datetime import datetime, time 
import json
import pandas as pd
import pyodbc


i = 0
data = []
with open('L:\\TRADING\SQLDataUpload\PET.txt') as f:
    for line in f:
        i = i + 1
        d = json.loads(line)
        df = pd.DataFrame(d)
        
        df['Date'] = [row[0] for row in df['data']]
        df['Newdate'] = [row+"01" if len(row) <8 else row for row in df['Date']]
        df['Amount'] = [row[1] for row in df['data']]

        
        connection = pyodbc.connect('''Driver={SQL Server}; 
                                    Server=STUKLS022; 
                                    Database=EIAData; 
                                    uid=GAMI ;
                                    pwd=pw;
                                    Trusted_Connection=yes''')    
        if "series_id" in df.columns:
            print (" Series Exists on line " + str(i))

            cursor = connection.cursor()
            SQLCommand = ("INSERT INTO Series"  
              "(id, Name, CategoryId, f, Units, updated2) "
              "VALUES (?,?,1,?,?,?)"
              )
            Values = list(zip(df.series_id,
                              df.name,
                              df.f,
                              df.units, 
                              pd.to_datetime(df.last_updated)))[0]
            
            cursor.execute(SQLCommand,Values)
           
            SQLCommand = ("INSERT INTO SeriesData"  
              "(series_id, Period, Value ) "
              "VALUES (?,?,?)"
              )
            Values = list(zip(df.series_id,  
                              pd.to_datetime(df['Newdate'], format='%Y%m%d'),
                              df['Amount'].astype(float)))
            
            cursor.executemany(SQLCommand,Values)
               
            
        else: 
            print ("Categories exist on line " + str(i))
            cursor = connection.cursor()
            SQLCommand = ("INSERT INTO Category"  
              "(id, Name, ParentID) "
              "VALUES (?,?,?)"
              )
            Values = list(zip(df.category_id, 
                              df.name, 
                              df.parent_category_id ))[0]
            
            cursor.execute(SQLCommand,Values)
    
connection.commit()
connection.close()