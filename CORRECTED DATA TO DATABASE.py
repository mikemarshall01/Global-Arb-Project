# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:37:29 2018

@author: mima
"""

from datetime import datetime, time 
import json
import pandas as pd
import pyodbc
import dateutil.parser
from pandas.io.json import json_normalize

df.loc[145]
category_id = df['category_id'][139168]

farm = df[df['series_id'] == 'PET.WCRFPUS2.W']
farm = df[df['childseries'] == 'PET.KDRVAFNUS1.A']
farm = df[df['parent_category_id'] == 235678]
farm = df[df['category_id'] == 235678]

df = pd.read_json('L:\\TRADING\SQLDataUpload\PET.txt', lines=True)

lol = json_normalize(df)

df['Date'] = [datetime.strptime((row[0]+"01"), '%Y%m%d').strftime('%d-%m-%Y')
                if len(row) <8 
                else datetime.strptime(row[0], '%Y%m%d').strftime('%d-%m-%Y')
                for row 
                in df['data']]
df['Amount'] = [float(row[1]) for row in df['data']]
df['last_updated'] = [dateutil.parser.parse(row).strftime('%d-%m-%Y') 
                                    for row in df['last_updated']]

connection = pyodbc.connect('''Driver={SQL Server}; 
                            Server=STCHGS112; 
                            Database=MIMAWorkSpace; 
                            uid=MIMA ;
                            pwd=pw;
                            Trusted_Connection=yes''')    

cursor = connection.cursor()

SQLCommand = (
            "INSERT INTO Series"  
            "(id, Name, CategoryId, f, Units, updated) "
            "VALUES (?,?,1,?,?,?)"
             )

Values = pd.concat([df.series_id,
                  df.name,
                  df.f,
                  df.units, 
                  df.last_updated], axis = 1).drop_duplicates().values.tolist()

cursor.executemany(SQLCommand,Values)
   
SQLCommand = (
            "INSERT INTO SeriesData"  
            "(series_id, Period, Value ) "
            "VALUES (?,?,?)"
              )

Values = pd.concat([df.series_id,  
                  df.Date,
                  df.Amount], axis = 1).drop_duplicates().values.tolist()

cursor.executemany(SQLCommand,Values)
    
SQLCommand = (
            "INSERT INTO Category"  
            "(id, Name, ParentID) "
             "VALUES (?,?,?)"
             )


Values = pd.concat([df.series_id, 
                  df.name, 
                  df.description], axis = 1).drop_duplicates().values.tolist()

cursor.executemany(SQLCommand,Values)

connection.commit()
connection.close()