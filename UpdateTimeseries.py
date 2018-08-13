# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:10:05 2018

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

data = pd.ExcelFile('C://Users//mima//Documents//price_freight_assay_data2.xlsx')

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

TSUpdateSQL = TSUpdateSQL()


def freight_rates_records_for_sql():
    raw_rates = pd.ExcelFile('C://Users//mima//Documents//flat_rates.xlsx')
    rates = []
    for x,y in enumerate([name.split()[2] for name in raw_rates.sheet_names]):
        f  = pd.read_excel(raw_rates, sheetname = x, header = None).iloc[1:47,1:]
        lplen = len(f.iloc[:,1])
        dplen = len(f.iloc[1,:])
        for j in range(1, dplen):
            for i in range(1,lplen):
                LoadPort = f.iloc[i,0]
                DischargePort = f.iloc[0,j]
                Year = y
                Rate = f.iloc[i,j]
                rates.append({'LoadPort':LoadPort, 'DischargePort': DischargePort, 'Year':Year,'Rate':Rate})    
    rate_data_input = pd.DataFrame(rates).dropna(axis=0).to_dict('records')
    return rate_data_input

def assay_records_for_sql():
    assay_input = pd.read_excel(data, 'assay', index_col='Database_Name')
    assay_input.dtypes
    assay_input['RESIDUE_v40'] = assay_input['RESIDUE_v40'].astype(float)
    assay_input['GradesId'] = assay_input['GradesId'].astype(float)
    assay_input = assay_input.reset_index()
    assay_input = assay_input.where(pd.notnull(assay_input),None).to_dict('records')
    return assay_input

def world_scale_records_for_sql():
    ws_input = pd.read_excel(data, 'ws').to_dict('records')
    return ws_input

def world_scale_mappings_data_for_sql():
    world_scale_mappings_input = pd.read_excel(data, 'sub_to_ws').to_dict('records')
    return world_scale_mappings_input

def exceptions_data_for_sql():
    exceptions = {
                'Arab Extra Light':
                    {'ROTTERDAM':{'Code':'AAIQQ00','Index':'BWAVE'},
                     'AUGUSTA':{'Code':'AAWQK00','Index':'BWAVE'},
                     'HOUSTON':{'Code':'AAIQZ00','Index':'WTI'},
                     'SINGAPORE':{'Code':'AAIQV00','Index':'OMAN/DUBAI'}},
                'Arab Light':
                    {'ROTTERDAM':{'Code':'AAIQR00','Index':'BWAVE'},
                    'AUGUSTA':{'Code':'AAWQL00','Index':'BWAVE'},
                    'HOUSTON':{'Code':'AAIRA00','Index':'WTI'},
                    'SINGAPORE':{'Code':'AAIQW00','Index':'OMAN/DUBAI'}},
                'Arab Medium':
                    {'ROTTERDAM':{'Code':'AAIQS00','Index':'BWAVE'},
                     'AUGUSTA':{'Code':'AAWQM00','Index':'BWAVE'},
                     'HOUSTON':{'Code':'AAIRB00','Index':'WTI'},
                     'SINGAPORE':{'Code':'AAIQX00','Index':'OMAN/DUBAI'}},
                'Arab Heavy':
                    {'ROTTERDAM':{'Code':'AAIQT00','Index':'BWAVE'},
                     'AUGUSTA':{'Code':'AAWQN00','Index':'BWAVE'},
                     'HOUSTON':{'Code':'AAIRC00','Index':'WTI'},
                     'SINGAPORE':{'Code':'AAIQY00','Index':'OMAN/DUBAI'}},
                'Basrah Light':
                    {'ROTTERDAM':{'Code':'AAIPH00','Index':'Dated'},
                     'AUGUSTA':{'Code':'AAIPH00','Index':'Dated'},
                     'HOUSTON':{'Code':'AAIPG00','Index':'WTI'},
                     'SINGAPORE':{'Code':'AAIPE00','Index':'OMAN/DUBAI'}},
                'Basrah Heavy':
                    {'ROTTERDAM':{'Code':'AAXUC00','Index':'Dated'},
                     'AUGUSTA':{'Code':'AAXUC00','Index':'Dated'},
                     'HOUSTON':{'Code':'AAXUE00','Index':'Mars'},
                     'SINGAPORE':{'Code':'AAXUA00','Index':'OMAN/DUBAI'}},
                'Iranian Heavy':
                    {'ROTTERDAM':{'Code':'AAIPB00','Index':'BWAVE'},
                     'AUGUSTA':{'Code':'AAUCH00','Index':'BWAVE'},
                     #'Iranian Heavy':{'HOUSTON':{'Code':abcde,'Index':'WTI'}},
                    'SINGAPORE':{'Code':'AAIOY00','Index':'OMAN/DUBAI'}},
                'Iranian Light':
                    {'ROTTERDAM':{'Code':'AAIPA00','Index':'BWAVE'},
                     'AUGUSTA':{'Code':'AAUCJ00','Index':'BWAVE'},
                    'SINGAPORE':{'Code':'AAIOX00','Index':'OMAN/DUBAI'}},
                'Forozan':
                    {'ROTTERDAM':{'Code':'AAIPC00','Index':'BWAVE'},
                    'AUGUSTA':{'Code':'AAUCF00','Index':'BWAVE'},
                    'SINGAPORE':{'Code':'AAIOZ00','Index':'OMAN/DUBAI'}},
                'Isthmus':{'ROTTERDAM':{'Code':'AAIQC00','Index':'Dated'},
                    'AUGUSTA':{'Code':'AAIQC00','Index':'Dated'},
                    'HOUSTON':{'Code':'AAIPZ00','Index':'WTI'},
                    'SINGAPORE':{'Code':'AAIQE00','Index':'OMAN/DUBAI'}},
                'Maya':{'ROTTERDAM':{'Code':'AAIQB00','Index':'Dated'},
                    'AUGUSTA':{'Code':'AAIQB00','Index':'Dated'},
                    'HOUSTON':{'Code':'AAIPY00','Index':'WTI'},
                    'SINGAPORE':{'Code':'AAIQD00','Index':'OMAN/DUBAI'}}
                }
      
    exceptions_table = pd.DataFrame.from_dict({(crude,destination): exceptions[crude][destination] 
            for crude in exceptions.keys() 
            for destination in exceptions[crude].keys()}, 
            orient='index')
    exceptions_input = exceptions_table.reset_index().rename(columns={'level_0':'Crude','level_1':'Destination'}).to_dict('records')  
    return exceptions_input

def wti_expiry():
    expiry_table = pd.read_excel(data, 'expiry', index_col = 'Date')
    expiry_table.index = pd.to_datetime(expiry_table.index)
    expiry_table = expiry_table.apply(pd.to_datetime, errors='coerce')
    expiry_table_input = expiry_table.unstack().reset_index().rename(columns={'level_0':'Series',0:'Value'}).to_dict('records')
    return expiry_table_input

def refinery_config_data_for_sql():
    refinery_configurations = {'simple':{'refinery_volume':200,
                             'reformer_capacity':42,
                             'fcc_capacity':48,
                             'coker_capacity':0,
                             'lvn_gasolinepool':0.12,
                             'kero_gasolinepool':0.15},
                    'complex':{'refinery_volume':350,
                             'reformer_capacity':100,
                             'fcc_capacity':80,
                             'coker_capacity':48,
                             'lvn_gasolinepool':0.12,
                             'kero_gasolinepool':0.15}} 
      
    refinery_configurations_table = pd.DataFrame.from_dict({ref: refinery_configurations[ref] 
            for ref in refinery_configurations.keys()}, 
            orient='index')
    refinery_configurations_input = refinery_configurations_table.unstack().reset_index().rename(columns={'level_0':'Series','level_1':'Refinery',0:'Value'}).to_dict('records')  
    return refinery_configurations_input

params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STUKLS022;DATABASE=Arbitrage;UID=mima;Trusted_Connection=Yes")
eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, echo=True)

Base = declarative_base()
Session = sessionmaker(bind=eng)
session = Session()

class Basrah_WS_Base(Base):
    """this tells SQLAlchemy that rows of Basrah_WS_Base table must be mapped to this class"""
    __tablename__ = 'Basrah_WS_Base'
    Id = Column(Integer, primary_key=True)
    Date = Column(Date)
    Series = Column(String(32))
    Value = Column(Float)
    
class Global_Flat_Rates(Base):
    __tablename__ = 'Global_Flat_Rates'
    #__table_args__ = {'extend_existing': True} 
    Id = Column(Integer, primary_key=True)
    DischargePort = Column(String(32))
    LoadPort = Column(String(32))
    Rate = Column(Float)
    Year = Column(Integer)
    
class Crude_Assay(Base):
    __tablename__ = 'Crude_Assay'
    __table_args__ = {'extend_existing': True} 
    Id = Column(Integer, primary_key=True)
    Database_Name = Column(String(32))
    H_Comet_Name = Column(String(32))
    Crude_Manager_Name = Column(String(32))
    Gravity = Column(Float)
    API = Column(Float)
    Sulphur = Column(Float)
    Conversion = Column(Float)
    LPG = Column(Float)
    LVN = Column(Float)
    HVN = Column(Float)
    KERO = Column(Float)
    LGO = Column(Float)
    HGO = Column(Float)
    VGO = Column(Float)
    RESIDUE = Column(Float)
    LGO_density = Column(Float)
    HGO_density = Column(Float)
    VGO_density = Column(Float)
    RESIDUE_density = Column(Float)
    LGO_sulphur = Column(Float)
    HGO_sulphur = Column(Float)
    VGO_sulphur = Column(Float)
    RESIDUE_sulphur = Column(Float)
    RESIDUE_v50 = Column(Float)
    RESIDUE_v40 = Column(Float)
    RESIDUE_v100 = Column(Float)
    GradesId = Column(Float)
    Code = Column(String(32))
    Index = Column(String(32))
    Basis = Column(String(32))
    LoadPort = Column(String(32))
    FOBLoadPort = Column(String(32))
    FOBCode = Column(String(32))

class World_Scale_Table(Base):
    __tablename__ = 'World_Scale_Table'
    __table_args__ = {'extend_existing': True} 
    Id = Column(Integer, primary_key=True)
    Name = Column(String(64))
    Origin = Column(String(32))
    Destination = Column(String(32))
    Size = Column(String(32))
    Volume = Column(Integer)
    Terms = Column(String(32))
    Code = Column(String(32))
    bbls = Column(Integer)
    
class World_Scale_Mappings(Base):
    __tablename__ = 'World_Scale_Mappings'
    Id = Column(Integer, primary_key=True)
    Port_SubRegion = Column(String(32))
    WS_Region = Column(String(32))
    local_index = Column(String(32))
    Price_Set = Column(String(32))
    
class Exceptions(Base):
    __tablename__ = 'Exceptions'
    Id = Column(Integer, primary_key=True)
    Crude = Column(String(32))
    Code = Column(String(32))    
    Destination = Column(String(32))
    Index = Column(String(32))

class Expiry_Table(Base):
    __tablename__ = 'Expiry_Table'
    __table_args__ = {'extend_existing': True} 
    Id = Column(Integer, primary_key=True)    
    Series = Column(String(32))
    Value = Column(Date)
    Date = Column(Date)
    
class Forties_Sulphur(Base):
    __tablename__ = 'Forties_Sulphur'
    Id = Column(Integer, primary_key=True)
    BuzzardContent = Column(Float)
    Date = Column(Date)
    
class Prices(Base):
    __tablename__ = 'Prices'
    Id = Column(Integer, primary_key=True)
    Series = Column(String)
    Code = Column(String(32))
    Date = Column(Date)
    Value = Column(Float)
    
class Trader_Assessed_Prices(Base):
    __tablename__ = 'Trader_Assessed_Prices'
    Id = Column(Integer, primary_key=True)
    Series = Column(String)
    Code = Column(String(32))
    Date = Column(Date)
    Value = Column(Float)

class Refinery_Configurations(Base):
    __tablename__ = 'Refinery_Configurations'
    Id = Column(Integer, primary_key=True)
    Series = Column(String)
    Refinery = Column(String)
    Value = Column(Float)

"""Create the tables"""    
Base.metadata.create_all(eng)
    
"""Commit the data to a database"""
session.bulk_insert_mappings(Basrah_WS_Base, basrah_ws_base_input)
session.bulk_insert_mappings(Global_Flat_Rates, freight_rates_records_for_sql())
session.bulk_insert_mappings(Crude_Assay, assay_records_for_sql())
session.bulk_insert_mappings(World_Scale_Table, world_scale_records_for_sql())
session.bulk_insert_mappings(World_Scale_Mappings, world_scale_mappings_data_for_sql())
session.bulk_insert_mappings(Exceptions, exceptions_input)
session.bulk_insert_mappings(Expiry_Table, expiry_table_input)
session.bulk_insert_mappings(Forties_Sulphur, forties_sulphur_input)
session.bulk_insert_mappings(Refinery_Configurations, refinery_configurations_input)


t = time.process_time()
prices_input = import_prices_prepare_for_sql()
crude_diffs_input = trader_assessed_diffs_for_sql()
print("prices prepared successfully: Time was {}".format(time.process_time() - t))


t2 = time.process_time()
session.bulk_insert_mappings(Prices, prices_input)
session.bulk_insert_mappings(Trader_Assessed_Prices, crude_diffs_input)

session.commit()
session.close()
print("prices imported successfully: Time was {}".format(time.process_time() - t2))
"""IF NEEDED TO UNDO"""
session.rollback()
