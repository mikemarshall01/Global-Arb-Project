# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:17:58 2018

@author: mima
"""
import pandas as pd
import pypyodbc
from datetime import timedelta
from datetime import datetime

data = pd.read_excel('M:\Med_History.xlsx', sheetname = 'Sheet1')
df = data.sort_values(by=['Vessel','Load Date'], ascending=True).reset_index(drop=True)

vcn = pypyodbc.connect('DRIVER=SQL Server Native Client 11.0;SERVER=STCHGS112;UID=gami;Trusted_Connection=Yes;APP=Microsoft Office 2010;WSID=STUKLW048;DATABASE=STG_Targo;')

vcn_query = '''
                SELECT distinct ShipDetails.Name Vessel, Ship.IMO IMO, ShipDetails.ValidUntil, Ship.ShipClass
                FROM STG_Targo.dbo.ShipDetails ShipDetails
                INNER JOIN Ship Ship
                ON Ship.Id = ShipDetails.IdShip
                ORDER BY Name
                '''
vcn_table= pd.read_sql(vcn_query, vcn)

df['stem_before'] = df['Load Date'].shift(1)
df['stem_after'] = df['Load Date'].shift(-1)
df['vessel_before'] = df['Vessel'].shift(1)
df['vessel_after'] = df['Vessel'].shift(-1)
unique_vessel = list(df['Vessel'].unique())

stems_list = []

for i in unique_vessel:
    temp_df = df.loc[df['Vessel']==i]
    #print(i)
    for index, row in temp_df.iterrows():
        #print(row)
        if( row['stem_after'] < (row['Load Date'] + timedelta(days=5))) and (row['vessel_after'] == (row['Vessel'])):
            stems_list.append(row['CARGO ID'])
        elif row['stem_before'] > (row['Load Date'] - timedelta(days=5)) and (row['vessel_before'] == (row['Vessel'])):
            stems_list.append(row['CARGO ID'])
            
mike = df[df['CARGO ID'].isin(stems_list)]

unique_vessel_list = list(df['Vessel'].unique())

vcn_table.loc[vcn_table['vessel'].isin(unique_vessel_list), ('vessel','shipclass')]





mike.loc[vcn_table]
mike = mike.rename(columns={'Vessel':'vessel'})
merged = pd.merge(mike, vcn_table[['vessel','shipclass']], on='vessel', how='left', indicator=True)

cols = ['Cargo ID', 'Grade','vessel','Quantity','shipclass','Load Date','Destination Location', 'Discharge Date','CARGO ID']
merged2 = merged[cols].sort_values(by=['vessel','Load Date'], ascending=True).reset_index(drop=True)

path='L:/TRADING/ANALYSIS/Python/MIMA/'
merged2.to_excel(path+'merged2.xlsx')



# =============================================================================
# Find vessel destnations and discharge dates
# =============================================================================


ClipperRawData = pd.read_excel("L:\TRADING\ANALYSIS\Clipper Ship Tracking Analysis\Clipper Data\Clipper Global Crude Full History.xlsm", sheetname='Data')
vessel_list = ['ABSHERON','Aegean Power','Antartic','Arctic','Astro Polaris','Azerbaijan','British Robin','Finesse','Kaveri Spirit',
                   'Kimbolos Warrior','Kriti Island','Kriti Samaria','Leo Sun','Maersk Pearl','Maratha','Minerva Antartica','Minerva Nounou',
                   'Myrtos','Nevsky Prospect','Nissos Schinoussa','Nissos Serifos','Ns Century','NS Columbus','Ottoman Integrity',
                   'Panagia Armata','Parthenon TS','Pissiotis','Seaprince','Seaprincess','Stride','Yasa Golden Marmara','Seabravery','NS Captain','Alterego ii','Miltiadis M ii',
                   'Alfa Alandia','Aegean Nobility','Dolviken','Alexia','Dubai Glamour','Absheron']

#clip_vessel_list = ClipperRawData['vessel'].str.lstrip()

filtered_clipper = ClipperRawData.loc[(ClipperRawData['grade'].isin(['AZERI LIGHT']))&
                                      (ClipperRawData['load_date'] > datetime(2017,5,1))&
                                      (ClipperRawData['load_date'] < datetime(2018,1,1)), ('vessel','load_date','offtake_port','offtake_country','offtake_date')]




