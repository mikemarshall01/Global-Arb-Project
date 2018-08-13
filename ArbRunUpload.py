# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:35:46 2018

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

class arb_model_output(object): 
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

    def compile_for_sql_and_generate_key(self): 
        self.global_arbs_sql = self.global_arbs.unstack().reset_index().rename(columns={0:'Value'}).dropna(how='any')
        #self.global_arbs_sql = self.global_arbs_sql.loc[self.global_arbs_sql['Date'] > date(2018,5,1)]
        self.global_arbs_sql['Unique_Key'] = self.global_arbs_sql['Date'].map(str) + self.global_arbs_sql['RefineryConfig'].map(str) +  self.global_arbs_sql['Grade'].map(str) + self.global_arbs_sql['Region'].map(str) +  self.global_arbs_sql['Series'].map(str)
        print("dataframe compiled and keys generated for upload: Time was {}. ".format(time.process_time() - self.t)) 
    
    def filter_for_new_only(self):
        query = Table('ArbModelOutput', self.meta, autoload=True)
        stm = select([query])  #.where(query.c.Date > (dt.today() - pd.Timedelta(days=20)))
        rs = self.con.execute(stm)
        TSdata = pd.DataFrame(rs.fetchall(), columns=rs.keys()).drop(['Id'], axis=1)
        TSdata['Unique_Key'] = TSdata['Date'].map(str) + TSdata['RefineryConfig'].map(str) +  TSdata['Grade'].map(str) + TSdata['Region'].map(str) +  TSdata['Series'].map(str)
        self.new_timeseries_records = self.global_arbs_sql[~self.global_arbs_sql['Unique_Key'].isin(TSdata['Unique_Key'])].drop(['Unique_Key'], axis=1).to_dict('records')  
        print("new records filtered and assigned to structure: Time was {}. Beginning upload... ".format(time.process_time() - self.t))
        
    def table_check_insert(self):     
        Base = declarative_base()
        Session = sessionmaker(bind=self.eng)
        session = Session()
        class ArbModelOutput(Base):
            """this tells SQLAlchemy that rows of Basrah_WS_Base table must be mapped to this class"""
            __tablename__ = 'ArbModelOutput'
            __table_args__ = {'extend_existing': True, 'schema':'dbo'} 
            Id = Column(Integer, primary_key=True)
            Date = Column(Date)
            Series = Column(String(32))
            Value = Column(Numeric(12,2))  
            RefineryConfig = Column(String(32))
            Grade = Column(String(32))
            Region = Column(String(32))
        if not self.eng.dialect.has_table(self.eng, 'ArbModelOutput', schema='dbo'):
            Base.metadata.create_all(self.eng)
        try:
            self.filter_for_new_only()
            session.bulk_insert_mappings(ArbModelOutput, self.new_timeseries_records)
            session.commit()
            session.close()
            print("Table checked and model output uploaded: Time was {}".format(time.process_time() - self.t)) 

        except Exception as e:
            print(e)
            session.rollback()  
            session.close()
    
# =============================================================================
#     temp = global_arbs.unstack().reset_index().rename(columns={0:'Value'}).dropna(how='any')
#     nan_df = temp[temp.isnull().any(axis=1)]
#     nan_df.head()
# =============================================================================
    
    def run(self):
        self.global_arbs = ArbRef_Compile.create_arb_matrix()
        print("global_arbs executed, attributes imported successfully: Time was {}".format(time.process_time() - self.t)) 
        self.compile_for_sql_and_generate_key()
        self.table_check_insert()
        
if __name__ == "__main__":
    #crude = 'AZERI'
    #destination = 'Houston'
    t5 = time.process_time()
    arb_model_output = arb_model_output()
    print("Time Series values updated successfully: Time was {}".format(time.process_time() - t5))