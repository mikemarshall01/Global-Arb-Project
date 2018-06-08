# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:58:51 2017

@author: mima
"""

import pyodbc
cnxn = pyodbc.connect(
        '''Driver={SQL Server Native Client 11.0};
        Server=STCHGS126;
        UID=gami;
        Trusted_Connection=yes;
        WSID=STUKLW05;
        DATABASE=STG_Price;a
        ServerSPN=STG_Price''')
cursor = cnxn.cursor()
cursor.execute("""SELECT 
               STUK_Prices.curve_num, 
               STUK_Prices.name, 
               STUK_Prices.asof_date, 
               STUK_Prices.maturity_date, 
               STUK_Prices.maturity_type, 
               STUK_Prices.price
               FROM 
               STG_Price.dbo.STUK_Prices STUK_Prices
               WHERE STUK_Prices.asof_date >= '1-jan-2010'""")
while 1:
    row = cursor.fetchone()
    if not row:
        break
    print(row)
cnxn.close()



import pyodbc
connection = pyodbc.connect('Driver={SQL Server};'
 'Server=STCHGS126;'
 'Database=STG_Price;'
 'uid=gami')
connection.close()