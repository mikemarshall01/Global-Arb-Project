# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 15:42:33 2018

@author: mima
"""
from sqlalchemy import create_engine, Table, MetaData, Integer, Float, String, Column, Date, Sequence, Numeric
from sqlalchemy.sql import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pyodbc
import pandas as pd
import urllib.parse
import time
from datetime import datetime as dt
from datetime import date
import ArbRef_Compile

"""
params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STUKLS022;DATABASE=Arbitrage;UID=mima;Trusted_Connection=Yes")
eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, echo=False)
meta = MetaData(eng, schema = 'dbo')
con = eng.connect()
query = Table('ArbYieldView', meta, autoload=True)
stm = select([query])
rs = con.execute(stm)
assay = pd.DataFrame(rs.fetchall(), columns=rs.keys())
string_cols = ['Code','Index','Basis','LoadPort','FOBLoadPort','FOBCode','Name']
numeric_cols = [x for x in rs.keys() if x not in string_cols]
assay[numeric_cols] = assay[numeric_cols].apply(pd.to_numeric, errors = 'coerce')
assay.index = assay['Name']
assay = assay.to_dict('index')
print("Assay imported successfully: Time was {}".format(time.process_time() - t))
  """

class PeckingOrder(object):
    """1. Creates the global arbs matrix
        2. Creates unique keys to filter for new entries
        3. Checks if arb model output table exists, if it doesn't, it creates it
        4. Runs the check on the table for duplicates, and only imports the none duplicates"""
    
    def __init__(self):
        self.t = time.process_time()
        self.params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STUKLS022;DATABASE=Arbitrage;UID=mima;Trusted_Connection=Yes")
        self.eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % self.params, echo=False)
        self.meta = MetaData(self.eng, schema = 'dbo')
        with self.eng.connect() as self.con:
            self.run() 
    
    def retrieve_data(self):
        query = Table('ArbModelOutput', meta, autoload=True)
        stm = select([query]).where(query.c.Date > (dt.today() - pd.Timedelta(days=40)))
        rs = con.execute(stm)
        TSdata = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        TSdata['Value'] = pd.to_numeric(TSdata['Value'])
        return TSdata
    
    def temp_for_mapping(self):
        query = Table('ArbWSDefView', meta, autoload=True)
        stm = select([query])
        rs = con.execute(stm)
        WSMap = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        WSMap['WS'] = WSMap['Size'].str[:4] + "_WS"
        mapping_ws = WSMap[['Name','WS']].set_index('Name').to_dict()
        TSdata['Series'] = TSdata['Series'].replace(mapping_ws['WS'])
        return TSdata
        
    def create_table(self):
        
        TSdata.set_index(['RefineryConfig','Grade','Region','Series','Date']).unstack(['RefineryConfig','Grade','Region','Series','Date'])
        TSdata.columns
        TSdata.set_index('Grade')
        TSdata.unstack(level=['RefineryConfig','Grade','Region','Series']).head()
        TSdata.loc[TSdata['Series'] == ']
        series_list = TSdata['Series'].unique()
        
        dtd_landed_comparison = manip.iloc[manip.index > date(2018,1,1),(manip.columns.get_level_values('RefineryConfig')=='simple')&
                 (manip.columns.get_level_values('Grade').isin(['Bakken','BaseBlendUSGC','Eagleford 45','LLS','Mars','Forties','Ekofisk','Saharan']))&
                 (manip.columns.get_level_values('Region')=='Rotterdam')]
        
        f_series_list = ['diff','Rate','Suez_WS', 'Suezmax','Costs','structure','Suez_landed_vs_dtd','GPW','Base_GPW']
        
        test_R = manip.iloc[manip.index > date(2018,1,1),
                                           (manip.columns.get_level_values('RefineryConfig')=='simple')&
                                           (manip.columns.get_level_values('Region')=='Rotterdam')&
                                           (manip.columns.get_level_values('Series').isin(f_series_list))]
        
        test_R.reset_index(level=['RefineryConfig','Region','Grade'])
        
        test_P = test_R.set_index(['RefineryConfig','Region','Grade'])

        manip.columns
        
        
        
        manip = TSdata.pivot_table(columns=['RefineryConfig','Grade','Region','Series'], index = 'Date', values='Value')
        manip.loc[:,(slice(None), slice(None), slice(None), ['TEXT'])] =  1
        manip.head()
        
        
        
        manip.loc[:,([:,:])]
        idx = pd.IndexSlice
        manip.loc[:,idx[:,:,:,f_series_list]]
        
        manip.loc[:,(:,:,:)]
        
        tdf = pd.DataFrame(manip).
        
        
        
        TSdata.pivot_table(columns=['Date','RefineryConfig','Out','Series'], index = 'Grade', values='Value')
        
        TSdata.pivot_table(columns=['Series'], index = 'Grade', values='Value')

        
        
        self.new_timeseries_records = self.global_arbs_sql[~self.global_arbs_sql['Unique_Key'].isin(TSdata['Unique_Key'])].drop(['Unique_Key'], axis=1).to_dict('records')  
        print("new records filtered and assigned to structure: Time was {}. Beginning upload... ".format(time.process_time() - self.t))
        
     def shape_data(self):  
         
        
        