# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:21:40 2018

@author: mima
"""

# =============================================================================
# Extract data from JODI database and amend place in table - issues were kbd vs kbbl on stocks so process extracts and joins creating a final table from 2002
# =============================================================================

import pypyodbc
import pandas as pd
idx = pd.IndexSlice

jodi_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=JodiOil;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')

#NWECountries = ['Belgium','Denmark','Finland','France','Germany','Ireland','Netherlands','Norway','Poland','Sweden','United Kingdom']
#NWECountriesCAP = list(map(lambda x: x.upper(), NWECountries))


jodi_query = """
               SELECT Countries.Country, PrimaryTable.Product, WorldPrimaryFLows.Description, PrimaryTable.Unit, CONVERT(datetime, CONCAT(PrimaryTable.Month,'-01'), 121) as Month, PrimaryTable.Quantity, PrimaryTable.StatusCode
               FROM JodiOil.dbo.PrimaryTable PrimaryTable 
               INNER JOIN JodiOil.dbo.Countries Countries ON PrimaryTable.Country = Countries.[Country Code]
               INNER JOIN JodiOil.dbo.WorldPrimaryFLows WorldPrimaryFLows ON PrimaryTable.Flow = WorldPrimaryFLows.[Code]
               WHERE PrimaryTable.Product in('CRUDEOIL','OTHERCRUDE')
               AND Countries.Country in ('Belgium','Denmark','Finland','France','Germany','Ireland','Lithuania','Netherlands','Norway','Poland','Sweden','United Kingdom')
               """               
               
jodi_table = pd.read_sql(jodi_query, jodi_connection)
jodi_table_kbbl = jodi_table[(jodi_table['unit'] == 'KBBL')]
jodi_table_kbd = jodi_table[(jodi_table['unit'] == 'KBD')]

jodi_pivot_kbd = pd.pivot_table(jodi_table_kbd, values = 'quantity', index = 'month', columns = ['country','description'], aggfunc='sum')
jodi_pivot_kbbl = pd.pivot_table(jodi_table_kbbl, values = 'quantity', index = 'month', columns = ['country','description'], aggfunc='sum')

jodi_pivot_kbbl.columns
jodi_pivot_kbd.columns

idx = pd.IndexSlice
jodi_kbd_data = jodi_pivot_kbd.loc[idx[:],idx[:,['Crude Production', 'Exports', 'Imports', 'Refinery Runs']]]
jodi_kbbl_data = jodi_pivot_kbbl.loc[idx[:],idx[:, ['Closing Stock Level', 'Stock Change']]]

NWE_Countries_crude = pd.concat([jodi_kbd_data, jodi_kbbl_data], axis=1)
NWE_Total_fund_crude = NWE_Countries_crude.groupby(level='description', axis=1).sum()


# =============================================================================
# For Products
# =============================================================================

jodi_products_query = """
                   SELECT Countries.Country, SecondaryTable.Product, WorldSecondaryFLows.Description, SecondaryTable.Unit, CONVERT(datetime, CONCAT(SecondaryTable.Date,'-01'), 121) as Month, SecondaryTable.Quantity, SecondaryTable.Code
                   FROM JodiOil.dbo.SecondaryTable SecondaryTable 
                   INNER JOIN JodiOil.dbo.Countries Countries ON SecondaryTable.Country = Countries.[Country Code]
                   INNER JOIN JodiOil.dbo.WorldSecondaryFLows WorldSecondaryFLows ON SecondaryTable.Flow = WorldSecondaryFLows.[Code]
                   WHERE Countries.Country in ('Belgium','Denmark','Finland','France','Germany','Ireland','Lithuania','Netherlands','Norway','Poland','Sweden','United Kingdom')
                   """    
                   
jodi_products_table = pd.read_sql(jodi_products_query, jodi_connection)
jodi_products_table_kbbl = jodi_products_table[(jodi_products_table['unit'] == 'KBBL')]
jodi_products_table_kbd = jodi_products_table[(jodi_products_table['unit'] == 'KBD')]

jodi_products_pivot_kbd = pd.pivot_table(jodi_products_table_kbd, values = 'quantity', index = 'month', columns = ['country','product','description'], aggfunc='sum')
jodi_products_pivot_kbbl = pd.pivot_table(jodi_products_table_kbbl, values = 'quantity', index = 'month', columns = ['country','product','description'], aggfunc='sum')

jodi_products_pivot_kbbl.columns
jodi_products_pivot_kbd.columns

jodi_products_kbd_data = jodi_products_pivot_kbd.loc[idx[:],idx[:,:,['Demand', 'Exports', 'Imports']]]
jodi_products_kbbl_data = jodi_products_pivot_kbbl.loc[idx[:],idx[:,:, ['Closing Stock Level', 'Stock Change']]]

NWE_Countries_products = pd.concat([jodi_products_kbd_data, jodi_products_kbbl_data], axis=1)
NWE_Total_fund_products = NWE_Countries_products.groupby(level='description', axis=1).sum()


# =============================================================================
# Now flaten the data in order to use in dataframe for testing
# =============================================================================
                   
NWE_Countries_products.columns = NWE_Countries_products.columns.to_series().str.join('_')                
#NWE_Total_fund_products.columns = NWE_Total_fund_products.columns.to_series().str.join('_')                      
NWE_Countries_crude.columns = NWE_Countries_crude.columns.to_series().str.join('_')                
#NWE_Total_fund_crude.columns = NWE_Total_fund_crude.columns.to_series().str.join('_')                      
                                      
path='L:/TRADING/ANALYSIS/Python/MIMA/'
NWE_Countries_products.to_excel(path+'NWE_Countries_products.xlsx')
NWE_Total_fund_products.to_excel(path+'NWE_Total_fund_products.xlsx') 
NWE_Countries_crude.to_excel(path+'NWE_Countries_crude.xlsx') 
NWE_Total_fund_crude.to_excel(path+'NWE_Total_fund_crude.xlsx')                    
                   
                   