# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:31:04 2018

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


class TSUpdateSQL(object):
    """This will load and insert the values for:
            Reuters Prices
            Trader Assessed Prices
            Forties Sulphur
            Basrah Conditions
            
        Connects to the database -> Loads prices etc from a sheet locally -> converts all data to
        dictionary format of Series, Value, Date and inserts"""
        
    def __init__(self):
        self.t = time.process_time()
        self.params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STUKLS022;DATABASE=Arbitrage;UID=mima;Trusted_Connection=Yes")
        self.eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % self.params, echo=False)
        self.data = pd.ExcelFile('C://Users//mima//Documents//price_freight_assay_data2.xlsx')
        self.trader_assessed = pd.ExcelFile('L://TRADING//ANALYSIS//GLOBAL//Arb Models//Pecking Order 2018.xlsm')
        self.meta = MetaData(self.eng, schema = 'dbo')
        with self.eng.connect() as self.con:
            self.run()         
        
    def latest_prices_and_generate_key(self):
        total = pd.read_excel(self.data, 'price_warehouse', header=4).drop(['Timestamp']).sort_index(ascending=True)
        total.index = pd.to_datetime(total.index)
        total.columns.name = 'Series'
        total.index.name = 'Date'
        total = total.unstack().reset_index().rename(columns={0:'Value'})
        total['Value'] = pd.to_numeric(total['Value'], errors='coerce')
        total = total.dropna(subset=['Value'])
        total['Source'] = 'REUTERS'
        total['Unique_Key'] = total['Date'].dt.date.map(str) + total['Series'].map(str)  + total['Source'].map(str)
        print("new prices compiled successfully")
        return total
    
    def latest_forties_and_generate_key(self):
        forties_sulphur = pd.read_excel(self.trader_assessed, 'Forties de-esc', header = [22], parse_cols="H:I").set_index('week ending')
        forties_sulphur = forties_sulphur.loc[pd.notnull(forties_sulphur.index)]
        forties_sulphur.index.name = 'Date'
        forties_sulphur_input = forties_sulphur.rename(columns={'buzzard content':'Value'})
        forties_sulphur_input['Value'] = pd.to_numeric(forties_sulphur_input['Value'], errors='coerce')
        forties_sulphur_input['Series'] = 'BuzzardContent'
        forties_sulphur_input['Source'] = 'SOCAR: FORTIES'
        forties_sulphur_input = forties_sulphur_input.reset_index()
        forties_sulphur_input['Unique_Key'] = forties_sulphur_input['Date'].dt.date.map(str) + forties_sulphur_input['Series'].map(str)  + forties_sulphur_input['Source'].map(str)
        print("new forties sulphur input compiled successfully")
        return forties_sulphur_input
    
    def latest_trader_assessed_and_generate_key(self):
        crude_diffs = pd.read_excel(self.trader_assessed, 'Crude Diffs Traders', header = 0)
        crude_diffs = crude_diffs.loc[pd.notnull(crude_diffs.index)]
        crude_diffs = crude_diffs.drop([name for name in crude_diffs.columns if 'Unnamed' in name], axis=1)
        crude_diffs.columns.name = 'Series'
        crude_diffs = crude_diffs.apply(pd.to_numeric, errors = 'coerce')
        crude_diffs.index.name = 'Date'
        crude_diffs.index = pd.to_datetime(crude_diffs.index)
        crude_diffs = crude_diffs.unstack().reset_index().rename(columns={0:'Value'})
        crude_diffs = crude_diffs.dropna(subset=['Value'])
        crude_diffs['Source'] = 'SOCAR: CRUDE DIFFS'
        crude_diffs['Unique_Key'] = crude_diffs['Date'].dt.date.map(str) + crude_diffs['Series'].map(str)  + crude_diffs['Source'].map(str)
        print("new trader assessed input compiled successfully")
        return crude_diffs
        
    def latest_basrah_and_generate_key(self):
        basrah_ws_base = pd.read_excel(self.data, 'basrah_ws_base', index_col = 'Date') 
        basrah_ws_base_input = pd.DataFrame(basrah_ws_base).unstack().reset_index().rename(columns={'level_0':'Series', 0:'Value'})
        basrah_ws_base_input['Source'] = 'SOCAR: BASRAH'
        basrah_ws_base_input['Unique_Key'] = basrah_ws_base_input['Date'].dt.date.map(str) + basrah_ws_base_input['Series'].map(str)  + basrah_ws_base_input['Source'].map(str)
        print("new basrah data input compiled successfully")
        return basrah_ws_base_input
    
    def compile_records(self):
        self.prices = self.latest_prices_and_generate_key()
        self.forties = self.latest_forties_and_generate_key()
        self.trader_assessed = self.latest_trader_assessed_and_generate_key()
        self.basrah = self.latest_basrah_and_generate_key()
        self.new_timeseries_records = pd.concat([self.prices, self.forties, self.trader_assessed, self.basrah])
        print("new records compiled successfully")
        
    def filter_for_new_only(self):
        query = Table('ArbTimeSeriesData', self.meta, autoload=True)
        stm = select([query])  #.where(query.c.Date > (dt.today() - pd.Timedelta(days=20)))
        rs = self.con.execute(stm)
        TSdata = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        TSdata['Unique_Key'] = TSdata['Date'].map(str) + TSdata['Series'].map(str)  + TSdata['Source'].map(str)        
        self.new_timeseries_records = self.new_timeseries_records[~self.new_timeseries_records['Unique_Key'].isin(TSdata['Unique_Key'])].drop(['Unique_Key'], axis=1).to_dict('records')  
    
    def table_check_insert(self):     
        Base = declarative_base()
        Session = sessionmaker(bind=self.eng)
        session = Session()
        class ArbTimeSeries(Base):
            """this tells SQLAlchemy that rows of Basrah_WS_Base table must be mapped to this class"""
            __tablename__ = 'ArbTimeSeriesData'
            __table_args__ = {'extend_existing': True, 'schema':'dbo'} 
            Id = Column(Integer, primary_key=True)
            Date = Column(Date)
            Series = Column(String(32))
            Value = Column(Numeric(12,2))  
            Source = Column(String(32))
        if not self.eng.dialect.has_table(self.eng, 'ArbTimeSeriesData', schema='dbo'):
            Base.metadata.create_all(self.eng)
        try:
            self.filter_for_new_only()
            session.bulk_insert_mappings(ArbTimeSeries, self.new_timeseries_records)
            session.commit()
            session.close()
        except Exception as e:
            print(e)
            session.rollback()  
            session.close()
        
    def run(self):
        self.compile_records()
        self.table_check_insert()
        print("Class methods executed, attributes imported successfully: Time was {}".format(time.process_time() - self.t)) 

if __name__ == "__main__":
    #crude = 'AZERI'
    #destination = 'Houston'
    t5 = time.process_time()
    TSUpdateSQL = TSUpdateSQL()
    print("Time Series values updated successfully: Time was {}".format(time.process_time() - t5))   
