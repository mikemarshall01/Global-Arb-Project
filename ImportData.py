# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:20:46 2018

@author: mima
"""

from sqlalchemy import create_engine, Table, MetaData, Column, Integer
from sqlalchemy.sql import select
import pyodbc
import pandas as pd
import urllib.parse
from datetime import datetime as dt
import time

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
  
class ImportData(object):
    """This is used to import the neccessary data from SQL so the ArbEconomics and RefineryEconomics programs can run.
    The data must be manipulated before it can be used - filling missing days, assigning one shot values over the month, etc.
    This creates the object that the files can work on, holding each neccessary dataframe in a different position"""
    
    def __init__(self):
        self.params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STUKLS022;DATABASE=Arbitrage;UID=mima;Trusted_Connection=Yes")
        self.eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % self.params, echo=False)
        self.meta = MetaData(self.eng, schema = 'dbo')
        with self.eng.connect() as self.con:
            self.run()         

    def get_assay(self):
        t = time.process_time()
        query = Table('ArbYieldView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        assay = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        string_cols = ['Code','Index','Basis','LoadPort','FOBLoadPort','FOBCode','Name']
        numeric_cols = [x for x in rs.keys() if x not in string_cols]
        assay[numeric_cols] = assay[numeric_cols].apply(pd.to_numeric, errors = 'coerce')
        assay.index = assay['Name']
        assay = assay.to_dict('index')
        print("Assay imported successfully: Time was {}".format(time.process_time() - t))
        return assay
    
    def get_rate_data(self):
        t = time.process_time()
        query = Table('ArbFlatRateView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        rate_data = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        print("Rate Data imported successfully: Time was {}".format(time.process_time() - t))
        return rate_data
    
    def get_sub_to_ws(self):
        t = time.process_time()
        query = Table('ArbMappingsView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        sub_to_ws = pd.DataFrame(rs.fetchall(), columns=rs.keys()).set_index('Port_SubRegion').to_dict()
        print("WS Mappings imported successfully: Time was {}".format(time.process_time() - t))
        return sub_to_ws
    
    def get_ws(self):
        t = time.process_time()
        query = Table('ArbWSDefView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        ws = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        print("World Scales imported successfully: Time was {}".format(time.process_time() - t))
        return ws
    
    def get_exceptions(self):
        t = time.process_time()
        query = Table('ArbMultiExceptionsView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        exceptions = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        print("Exceptions imported successfully: Time was {}".format(time.process_time() - t))
        return exceptions
    
    def get_timeseries(self):
        t = time.process_time()
        query = Table('ArbTimeSeriesData', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        TSdata = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        TSdata['Value'] = pd.to_numeric(TSdata['Value'])
        total = TSdata.loc[TSdata['Source'] == 'REUTERS'].drop(['Source'], axis=1)
        
        total.loc[total.duplicated(keep='last')]
        
        total = total.pivot(columns='Series', index = 'Date', values='Value').sort_index(ascending=True)
        total.index = pd.to_datetime(total.index)
        total.fillna(method='ffill', inplace=True)
        if total.index[-1] - total.index[-2] > pd.Timedelta(days=2):
            total.index.values[-1] = total.index[-2] + pd.Timedelta(days=1)
        
        total['AALSM01'].loc[total['AALSM01'] > 30] = total['AALSM01'].loc[total['AALSM01'] > 30] - total['CLc1']
        cfd_list = ['PCAKA00','PCAKC00','PCAKE00','PCAKG00','AAGLU00','AAGLV00','AALCZ00','AALDA00']
        temp = total[cfd_list].sub(pd.Series(total['PCAAQ00'] - total['PCAAR00']), axis=0)
        temp = temp[temp.index > dt(2017,6,30)]
        total.loc[total.index.isin(temp.index), list(temp.columns)] = temp[list(temp.columns)]
        
        crude_diffs = TSdata.loc[(TSdata['Source'] == 'SOCAR: CRUDE DIFFS')].drop(['Source'], axis=1)
        crude_diffs = crude_diffs.pivot(columns='Series', index = 'Date', values='Value').sort_index(ascending=True)
        crude_diffs = crude_diffs.reindex(total.index).fillna(method='bfill').fillna(method='ffill')
        total.update(crude_diffs[[x for x in crude_diffs.columns if x not in ['Basrah Light', 'Basrah Heavy']]])

        forties_sulphur = TSdata.loc[(TSdata['Source'] == 'SOCAR: FORTIES')].set_index('Date').drop(['Source'], axis=1)            
        forties_sulphur = forties_sulphur.loc[pd.notnull(forties_sulphur.index)]
        forties_sulphur = forties_sulphur.reindex(total.index).fillna(method='ffill').fillna(method='bfill')
        
        basrah_ws_base = TSdata.loc[(TSdata['Source'] == 'SOCAR: BASRAH')].drop(['Source'], axis=1) 
        basrah_ws_base = basrah_ws_base.pivot(columns='Series', index = 'Date', values='Value').sort_index(ascending=True)
        basrah_ws_base.index = pd.to_datetime(basrah_ws_base.index)
              
        query = Table('ArbMultiExceptionsView', self.meta, autoload=True)
        stm = select([query.c.Code])
        rs = self.con.execute(stm)
        crudes_with_osps = pd.DataFrame(rs.fetchall(), columns=rs.keys())['Code'].tolist()
        total[crudes_with_osps] = total[crudes_with_osps].resample('MS').mean().shift(-1, freq='MS').reindex(total.index).fillna(method='ffill')
        
        query = Table('Expiry_Table', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        expiry_table = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        expiry_table = expiry_table.pivot(columns='Series', index = 'Date', values='Value').sort_index(ascending=True)
        expiry_table.index = pd.to_datetime(expiry_table.index, errors='coerce')
        expiry_table.apply(pd.to_datetime, errors='coerce')            
        df = pd.DataFrame(index=total.index)
        df['Date'] = df.index
        for_dates = lambda x: (expiry_table.loc[(expiry_table.index.month == x.month)&(expiry_table.index.year == x.year)]['WTI_Expiry']).iat[0]
        df['WTI_Expiry'] = df['Date'].apply(for_dates)
        df.drop(['Date'], inplace=True, axis=1)
        print("Prices formatted and imported successfully: Time was {}".format(time.process_time() - t))
        return total, crude_diffs, forties_sulphur, df, basrah_ws_base
    
    def get_ports(self):
        t = time.process_time()        
        query = Table('TargoRegionArbView', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        ports = pd.DataFrame(rs.fetchall(), columns=rs.keys())
        print("Ports imported successfully: Time was {}".format(time.process_time() - t))
        return ports
    
    def get_refinery_configs(self):
        t = time.process_time()
        query = Table('Refinery_Configurations', self.meta, autoload=True)
        stm = select([query])
        rs = self.con.execute(stm)
        refinery_configs = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        refinery_configs = refinery_configs.pivot(columns='Series', index = 'Refinery', values='Value').to_dict(orient='index')
        print("Refinery_Configurations imported successfully: Time was {}".format(time.process_time() - t))
        return refinery_configs
    
    def run(self):
        t2 = time.process_time()
        self.assay = self.get_assay()
        self.rate_data = self.get_rate_data()
        self.sub_to_ws = self.get_sub_to_ws()
        self.ws = self.get_ws()
        self.exceptions = self.get_exceptions()
        self.total, self.crude_diffs, self.forties_sulphur, self.df, self.basrah_ws_base = self.get_timeseries()
        self.ports = self.get_ports()
        self.refinery_configs = self.get_refinery_configs()
        print("Class methods executed, attributes imported successfully: Time was {}".format(time.process_time() - t2))
        
"""This is for debugging"""
if __name__ == "__main__":
    t5 = time.process_time()
    arb_data= ImportData()
    crude = 'Azeri'
    destination = 'Houston'  
    print("ArbData created successfully: Time was {}".format(time.process_time() - t5))
    
