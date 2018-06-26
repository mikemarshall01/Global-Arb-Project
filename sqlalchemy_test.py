# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:10:05 2018

@author: mima
"""

from sqlalchemy import create_engine, Table, MetaData, Integer, Float, String, Column, Date, Sequence
from sqlalchemy.sql import select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import sqlite3
import pyodbc
import pandas as pd
import urllib.parse
import time


data = pd.ExcelFile('C://Users//mima//Documents//price_freight_assay_data.xlsx')

"""PRICES"""

"""this new total table generates all the prices in one place for us"""




    df1.columns = 

pd.DataFrame.from_records(database_prices, columns = ['Series','Code'])

database_prices.stack().stack(level=1)

database_prices


m_index = pd.MultiIndex.from_tuples(tuples, names=['name', 'code'])

df1 = database_prices.pivot(columns='Code', index = 'Date', values='Value')


database_prices.groupby(['Series','Code']).mean()
database_prices.dtypes
df1.columns = []



def create_price_table():
    

total.head()



where(pd.notnull(assay_input),None).to_dict('records')


total.index = pd.to_datetime(total.index)
total.sort_index(inplace=True)
total.fillna(method='ffill', inplace=True)
total = total[total.index > dt(2015,1,1)]


def import_prices_prepare_for_sql():
    total = pd.read_excel(data, 'price_warehouse', header=4).drop(['Timestamp']).iloc[:10]
    total_descriptions = pd.read_excel(data, 'price_warehouse', header=3).columns
    total.columns = [total_descriptions,total.columns]
    total.index = pd.to_datetime(total.index)
    total.columns.names = ('Series','Code')
    total.index.name = 'Date'
    total = total.unstack().reset_index().rename(columns={0:'Value'})
    total['Value'] = pd.to_numeric(total['Value'], errors='coerce')
    total = total.dropna(subset=['Value'])
    prices_input = total.to_dict('records')
    return prices_input

def basrah_data_records():
    basrah_ws_base = pd.read_excel(data, 'basrah_ws_base', index_col = 'Date') 
    basrah_ws_base_input = pd.DataFrame(basrah_ws_base).unstack().reset_index().rename(columns={'level_0':'Series', 0:'Value'}).to_dict('records') 
    return basrah_ws_base_input

def freight_rates_records():
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

def assay_records():
    assay_input = pd.read_excel(data, 'assay', index_col='Database_Name')
    assay_input.dtypes
    assay_input['RESIDUE_v40'] = assay_input['RESIDUE_v40'].astype(float)
    assay_input['GradesId'] = assay_input['GradesId'].astype(float)
    assay_input = assay_input.reset_index()
    assay_input = assay_input.where(pd.notnull(assay_input),None).to_dict('records')
    return assay_input

def world_scale_records():
    ws_input = pd.read_excel(data, 'ws').to_dict('records')
    return ws_input

def world_scale_mappings_data():
    world_scale_mappings_input = pd.read_excel(data, 'sub_to_ws').to_dict('records')
    return world_scale_mappings_input

def exceptions_data():
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
      
    exceptions_input = pd.DataFrame.from_dict({(crude,destination): exceptions[crude][destination] 
            for crude in exceptions.keys() 
            for destination in exceptions[crude].keys()}, 
            orient='index')
    exceptions_input = exceptions_input.unstack().unstack().reset_index().rename(columns={'level_0':'Series','level_1':'Destination','level_2':'Crude', 0:'Value'}).to_dict('records')  
    return exceptions_input

def forties_sulphur_records():
    trader_assessed = pd.ExcelFile('L://TRADING//ANALYSIS//GLOBAL//Arb Models//Pecking Order 2018.xlsm')
    forties_sulphur = pd.read_excel(trader_assessed, 'Forties de-esc', header = [22], parse_cols="H:I").set_index('week ending')
    forties_sulphur = forties_sulphur.loc[pd.notnull(forties_sulphur.index)]
    forties_sulphur_input = forties_sulphur.reset_index().rename(columns={'buzzard content':'BuzzardContent','week ending':'Date'}).to_dict('records')
    return forties_sulphur_input



params = urllib.parse.quote("DRIVER={SQL Server Native Client 11.0};SERVER=STCHGS112;DATABASE=MIMAWorkSpace;UID=mima;Trusted_Connection=Yes")
eng = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params, echo=True)

Base = declarative_base()
session = Session(bind=eng)

class Basrah_WS_Base(Base):
    """this tells SQLAlchemy that rows of Basrah_WS_Base table must be mapped to this class"""
    __tablename__ = 'Basrah_WS_Base'
    Id = Column(Integer, primary_key=True)
    Date = Column(Date)
    Series = Column(String(32))
    Value = Column(Float)
    
class Global_Flat_Rates(Base):
    __tablename__ = 'Global_Flat_Rates'
    __table_args__ = {'extend_existing': True} 
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
    HGO_desnsity = Column(Float)
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
    Destination = Column(String(32))
    Series = Column(String(32))
    Value = Column(String(32))
    
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




    

"""Create the tables"""    
Base.metadata.create_all(eng)
    
"""Commit the data to a database"""
session.bulk_insert_mappings(Basrah_WS_Base, basrah_ws_base_input)
session.bulk_insert_mappings(Global_Flat_Rates, rate_data_input)
session.bulk_insert_mappings(Crude_Assay, assay_input)
session.bulk_insert_mappings(World_Scale_Table, ws_input)
session.bulk_insert_mappings(World_Scale_Mappings, world_scale_mappings_input)
session.bulk_insert_mappings(Exceptions, exceptions_input)
session.bulk_insert_mappings(Forties_Sulphur, forties_sulphur_input)
session.bulk_insert_mappings(Prices, prices_input)




session.commit()


session.rollback()



Base.metadata.bind = eng

DBSession = sessionmaker(bind=eng)



"""EXTRACTION: 
    Create the connection to the database to be able to upload data"""

with eng.connect() as con:
    
    """Load definition of the Basrah_WS_Base table and the connection metadata"""
    meta = MetaData(eng)
    basrah_ws_base = Table('Basrah_WS_Base', meta, autoload=True)
    
    """Create the SQL select statement and execute and print - use table.c.column name for individual columns"""
    stm = select([basrah_ws_base])
    rs = con.execute(stm) #You can also put a standard query in here
    data = pd.DataFrame(rs.fetchall(), columns=rs.keys())
    df = data.pivot(columns='Series', index = 'Date', values='Value')
    print(df)
    
    
    

"""Import data from local files if needed"""
raw_rates = pd.ExcelFile('C://Users//mima//Documents//flat_rates.xlsx')
trader_assessed = pd.ExcelFile('L://TRADING//ANALYSIS//GLOBAL//Arb Models//Pecking Order 2018.xlsm')

assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
ws = pd.read_excel(data, 'ws')
expiry_table = pd.read_excel(data, 'expiry', index_col = 'Month')
ports = pd.read_excel(data, 'ports')
sub_to_ws = pd.read_excel(data, 'sub_to_ws', header = None)
sub_to_ws = sub_to_ws.set_index([0]).to_dict()


 



engine.execute("SELECT Series, Date, Value FROM Basrah_WS_Base").fetchall()

pd.DataFrame(engine.execute("SELECT Series, Date, Value FROM Basrah_WS_Base").fetchall())  
basrah_ws_base = basrah_ws_base_flat.pivot(columns='Series', index='Date', values='Value')





test = pd.DataFrame(refinery_configurations).unstack().reset_index().rename(columns={'index':'configuration'})
pd.DataFrame(refinery_configurations).T.reset_index()

t = time.process_time()
test.to_sql('Refinery_Configs_2', con =engine, index_label = 'Id', if_exists = 'replace')
print("Uploaded successfully: Time was {}".format(time.process_time() - t))

test.to_csv('L:/TRADING/ANALYSIS/Python/test.csv', sep='\t')




def retrieve_prices_model():
    df1 = database_prices.pivot(columns='Code', index = 'Date', values='Value')


















import pandas as pd
import numpy as np
from datetime import datetime as dt
import time
from pandas.tseries.offsets import BDay


    
def import_data():
    t2 = time.process_time()

    data = pd.ExcelFile('C://Users//mima//Documents//price_freight_assay_data.xlsx')
    raw_rates = pd.ExcelFile('C://Users//mima//Documents//flat_rates.xlsx')
    trader_assessed = pd.ExcelFile('L://TRADING//ANALYSIS//GLOBAL//Arb Models//Pecking Order 2018.xlsm')
    
    assay = pd.read_excel(data, 'assay', index_col = 'Database_Name').to_dict('index')
    ws = pd.read_excel(data, 'ws')
    expiry_table = pd.read_excel(data, 'expiry', index_col = 'Month')
    ports = pd.read_excel(data, 'ports')
    sub_to_ws = pd.read_excel(data, 'sub_to_ws', header = None)
    sub_to_ws = sub_to_ws.set_index([0]).to_dict()
    
    """table containing the basrah base worldscale that they fix their freight against"""
    basrah_ws_base = pd.read_excel(data, 'basrah_ws_base', index_col = 'YEAR')
    
    
    
    """Take in the crude prices and codes and convert to a dataframe.
    We need to take the first 2 rows of the prices with no headers as this will give us the cude name and the code ascociated
    Then transpose from rows to columns and rename the columns. This will be for later when we determine crude prices basis desired comaprison"""
    #prices_reference = (pd.read_excel(data, 'paper prices', header = None).iloc[0:2,1:]).transpose().rename(columns={0:'Name', 1: 'Code'})  
    
    """Merge the WS table with the prices table, slice df so 2016 onwards (Flat rates last date is 2015). 
    We don't drop rows now as dropping would be dependent on any nans in any column"""
    #total = prices.merge(ws_table, how = 'inner', left_index = True, right_index = True)
    #total = total.merge(paper_prices, how = 'inner', left_index = True, right_index = True)
    #total = total.iloc[total.index > dt(2015,12,31)]
    
    """this new total table generates all the prices in one place for us"""
    total = pd.read_excel(data, 'price_warehouse', header = 4).drop(['Timestamp'])
    total.index = pd.to_datetime(total.index)
    total.sort_index(inplace=True)
    total.fillna(method='ffill', inplace=True)
    total = total[total.index > dt(2015,1,1)]
    
    """We know there are some perculiarities in the data, such as the OSPs. So create this table here to handle. Found out need to shift the prices back a month but in order
    to identify which ones, needed the list of OSP crudes"""
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
     
    crudes_to_shift = pd.DataFrame.from_dict({(crude,destination): exceptions[crude][destination] 
            for crude in exceptions.keys() 
            for destination in exceptions[crude].keys()}, 
            orient='index')
    
    """convert the dataseries to a list, then use setr to get the unique items, then convert back to a list"""   
    crudes_to_shift = list(set(list(crudes_to_shift['Code'])))
    
    """Fopr the crudes in the list, I want to resample the series at the month start so there is a common value for the start of each month,
    I then want shift these values by 1 backwards, in this case because we resampled, this automatically means shift abck one month,
    I then want to re-index the new dataframe to conform to where we are putting it back into, and finally I assign the total dataframe where the 
    column headers are equal to the crude list, the new shifted and filled forward values to make sure everything lines up"""
    total[crudes_to_shift] = total[crudes_to_shift].resample('MS').mean().shift(-1, freq='MS').reindex(total.index).fillna(method='ffill')  

    #total['AAXUC00']
    
    """This will help with the date error. Turn the index into a numpy array and then assign the value"""
    if total.index[-1] - total.index[-2] > pd.Timedelta(days=2):
        total.index.values[-1] = total.index[-2] + pd.Timedelta(days=1)


    """Clean the column hedaers so no white spcaes - use simple list comprehension and set headers equal to cleaned"""
    cleaned_column_headers = [i.strip() for i in total.columns.values]
    total.columns = cleaned_column_headers
    
    """The below was get rid of the row in the index that hax NaT against it and then expand to daily and fill backwards"""
    crude_diffs = pd.read_excel(trader_assessed, 'Crude Diffs Traders', header = 0)
    crude_diffs = crude_diffs.loc[pd.notnull(crude_diffs.index)]
    crude_diffs = crude_diffs.drop([name for name in crude_diffs.columns if 'Unnamed' in name], axis=1)

   
    #crude_diffs.index = crude_diffs.index.map(lambda x : x + 1*BDay())
    crude_diffs = crude_diffs.reindex(total.index).fillna(method='bfill').fillna(method='ffill')
    
    """Slice the crude diffs where the dates in the index are the same as the dates in the total dataframe"""
    #crude_diffs = crude_diffs[crude_diffs.index.isin(total.index)]
    crudes_diff_against_osp = ['Basrah Light','Basrah Heavy']
    codes_list = [x for x in crude_diffs.columns if x not in crudes_diff_against_osp]
    
    """Apply the values in crude diffs to the correct codes and dates in the total dataframe"""
    total.update(crude_diffs[codes_list])
    
    
        
    
    """We have to convert the prices that are in absolutes into a diff vs a local index, and if there are, set to zero.
    This is LOOP Sour"""
    total['AALSM01'].loc[total['AALSM01'] > 30] = total['AALSM01'].loc[total['AALSM01'] > 30] - total['CLc1']
    #total.loc[total.index.isin(crude_diffs.index), codes_list] = crude_diffs[codes_list]
    #total[codes_list]
    
    #total.update(crude_diffs[codes_list])
    """ Need this for the sulphur table"""
    forties_sulphur = pd.read_excel(trader_assessed, 'Forties de-esc', header = [22], parse_cols="H:I").set_index('week ending')
    forties_sulphur = forties_sulphur.loc[pd.notnull(forties_sulphur.index)]
    forties_sulphur = forties_sulphur.reindex(total.index).fillna(method='ffill')

    """Also need to adjust the cfds to take into account the inter month BFOE spread"""   
    cfd_list = ['PCAKA00','PCAKC00','PCAKE00','PCAKG00','AAGLU00','AAGLV00','AALCZ00','AALDA00']
    temp = total[cfd_list].sub(pd.Series(total['PCAAQ00'] - total['PCAAR00']), axis=0)
    temp = temp[temp.index > dt(2017,6,30)]
    total.loc[total.index.isin(temp.index), list(temp.columns)] = temp[list(temp.columns)]
    
    """This turns the 5 years of rate matricies into a table for use to reference - 12/04/2018"""    
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
            
    rate_data = pd.DataFrame(rates)
    
    """Also initialise the temp df with index of total. Temp df is tol hold the dataseries needed to calculate the freight"""
    df = pd.DataFrame(index=total.index)
    df['Date'] = df.index
    
    """This function allows us to apply the expiration date for the wti futures used to determine what structure we apply to the CMA
    Have tried timing and slight improvment with the blow of 0.2seconds...."""
   
    t = time.process_time()

    for_dates = lambda x: (expiry_table.loc[(expiry_table.index.month == x.month)&(expiry_table.index.year == x.year)]['Expiry']).iat[0]
   
    df['Expiry'] = df['Date'].apply(for_dates)
    df.drop(['Date'], inplace=True, axis=1)
    
    
    
    

    print("df['Expiry'] created successfully: Time was {}".format(time.process_time() - t))
    print("Temp DataFrame created successfully")
    print("import_data() created successfully: Time was {}".format(time.process_time() - t2))
    
    return assay, ws, ports, total, rate_data, sub_to_ws, df, basrah_ws_base, crude_diffs, forties_sulphur, exceptions, crudes_to_shift





































cxn = pyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                    'Server=STCHGS112;'
                                    'Database=MIMAWorkSpace;'
                                    'uid=mima;'
                                    'Trusted_Connection=Yes;')
    
query = '''CREATE TABLE Global_Arbs_GPWs (
                ID int IDENTITY (1,1) PRIMARY KEY,
                RefineryConfig varchar(255) NOT NULL,
                Grade varchar(255) NOT NULL,
                Region varchar(255) NOT NULL,
                Series varchar(255) NOT NULL,
                Date DATE NOT NULL,
                Value FLOAT) '''