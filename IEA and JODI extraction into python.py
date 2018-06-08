# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 12:41:06 2018

@author: mima
"""

import pypyodbc
import pandas as pd
from datetime import datetime

iea_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=IEAData;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')

jodi_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=JodiOil;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')


iea_query = """
                SELECT CrudeData.Country, CrudeData.Product, Balance.[Desc], CrudeData.Period, CrudeData.Quantity, CrudeData.Asof, CrudeData.PeriodType
                FROM IEAData.dbo.CrudeData CrudeData
                INNER JOIN IEAData.dbo.Balance Balance ON CrudeData.Balance = Balance.Flow
                ORDER BY CrudeData.Country
                """
jodi_query = """
               SELECT Countries.Country, PrimaryTable.Product, WorldPrimaryFLows.Description, PrimaryTable.Unit, CONVERT(datetime, CONCAT(PrimaryTable.Month,'-01'), 121) as Month, PrimaryTable.Quantity, PrimaryTable.StatusCode
               FROM JodiOil.dbo.PrimaryTable PrimaryTable 
               INNER JOIN JodiOil.dbo.Countries Countries ON PrimaryTable.Country = Countries.[Country Code]
               INNER JOIN JodiOil.dbo.WorldPrimaryFLows WorldPrimaryFLows ON PrimaryTable.Flow = WorldPrimaryFLows.[Code]
               """               
               

iea_table = pd.read_sql(iea_query, iea_connection)
jodi_table = pd.read_sql(jodi_query, jodi_connection)

jodi_table_kbbl = jodi_table[(jodi_table['unit'] == 'KBBL') & (jodi_table['product'].isin(['CRUDEOIL','OTHERCRUDE']))]

jodi_table_kbd = jodi_table[(jodi_table['unit'] == 'KBD') & (jodi_table['product'].isin(['CRUDEOIL','OTHERCRUDE']))]

jodi_pivot_kbd = pd.pivot_table(jodi_table_kbd, values = 'quantity', index = 'month', columns = 'description', aggfunc='sum')


jodi_pivot_kbbl = pd.pivot_table(jodi_table_kbbl, values = 'quantity', index = 'month', columns = 'description', aggfunc='sum')

world_sd = pd.concat([jodi_pivot_kbd[['Crude Prodction',
                                      'Refinery Runs',
                                      'Exports',
                                      'Imports',
                                      'Other sources',
                                      'Statistical Difference',
                                      'Products Transferred/Backfloes',
                                      'Direct Use']], 
                                    jodi_pivot_kbbl[['Closing Stock Level','Stock Change']]], axis=1)




jodi_table_kbd_0809 = jodi_table_kbd[(jodi_table_kbd['month'] > datetime(2007, 11, 1)) &
                                     (jodi_table_kbd['month'] < datetime(2012, 2, 1)) &
                                     (jodi_table_kbd['product'] == 'TOTCRUDE') &
                                     (jodi_table_kbd['description'] == 'Crude Prodction')
                                     ]

jodi_table_kbd_final = jodi_table_kbd[(jodi_table_kbd['product'].isin(['CRUDEOIL','OTHERCRUDE','NGL']))]

jodi_table_kbbl_final = jodi_table_kbbl[(jodi_table_kbbl['product'] == 'TOTCRUDE') &
                                     (jodi_table_kbbl['description'] == 'Crude Prodction')
                                     ]

####
jodi_table_kbd_0809_nozeros = jodi_table_kbd_0809.loc[(jodi_table_kbd_0809 != 0).all(axis=1), :]

###jodi_table_kbd_0809 = jodi_table_kbd[(jodi_table_kbd['month'].dt.strftime('%m/%d/%Y') > '01/01/2009')]

jodi_pivot_kbbl = pd.pivot_table(jodi_table_kbbl, values = 'quantity', index = 'month', columns = ['description','country'], aggfunc='sum')
jodi_pivot_kbd = pd.pivot_table(jodi_table_kbd_final, values = 'quantity', index = 'month', columns = ['description'], aggfunc='sum')

list(jodi_table['description'].unique())

len(list(jodi_table['country'].unique()))

list(iea_table['country'].unique())

list(jodi_table['month'].unique())

list(jodi_table['product'].unique())

print(jodi_table.head())

cursor = connection.cursor()

cursor.execute("""
               SELECT Supply.Country [Country], Supply.Product, Supply.Period, Supply.Quantity, Supply.Asof, Supply.PeriodType
               FROM IEAData.dbo.Supply Supply
               WHERE (Supply.Country In ('USA')) 
               AND Supply.PeriodType = 'MTH'
               AND Supply.Asof = (Select MAX(Asof) FROM  IEAData.dbo.Supply)
               AND Supply.Product IN ('CRUDE', 'COND')
               AND Period >= '1-jan-14'
               ORDER BY Supply.Country
               """)

tables = cursor.fetchall()

pd.read_sql("""
               SELECT Supply.Country [Country], Supply.Product, Supply.Period, Supply.Quantity, Supply.Asof, Supply.PeriodType
               FROM IEAData.dbo.Supply Supply
               WHERE (Supply.Country In ('USA')) 
               AND Supply.PeriodType = 'MTH'
               AND Supply.Asof = (Select MAX(Asof) FROM  IEAData.dbo.Supply)
               AND Supply.Product IN ('CRUDE', 'COND')
               AND Period >= '1-jan-14'
               ORDER BY Supply.Country
               """)

for row in tables:
    print(row)

IEA_test_list = []
for row in tables:
    IEA_test_list.append({'Country':row.Country, 'Product':row.Product, 'Period':row.Period, 'Asof':row.Asof, 'PeriodType':row.PeriodType})
IEAtest = pd.DataFrame(IEA_test_list)
    
print(tables)




jodi_connection.close()

iea_connection.close()