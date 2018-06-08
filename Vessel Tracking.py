# -*- coding: utf-8 -*-
"""
Created on Wed May 23 15:53:44 2018

@author: mima
"""



""" Aim is see for each taker of Iranian:
    1) what volumes they take 
    2) what they subbed with during sanctions
    3) what I think they will replace the volumes with
    """


import pyodbc
import pandas as pd


cxn = pyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STCHGS126;'
                                'Database=STG_Targo;'
                                'uid=mima;'
                                'Trusted_Connection=Yes;')

query = "Select * From view_CargoAnalysis_External_Crude_Kb"

flows_data = pd.read_sql(query, cxn)
flows = flows_data.pivot_table(columns='LoadDate', index=['DischargeRegion','DischargeCountry','CurrentOwner','CleanGrade'], values = 'CleanQuantity')

flows_data.dtypes

"""slice the flows to only seen iranian exports"""
iran_raw_exports = flows_data[flows_data['LoadCountry'] == 'Iran']

"""turn datframe into pivot table with region, country, owner where possible and grade granularity"""
iran_exports = iran_raw_exports.pivot_table(columns='LoadDate', index=['DischargeRegion','DischargeCountry','CurrentOwner','CleanGrade'], values = 'CleanQuantity')
iran_countries = iran_exports.groupby(['DischargeRegion','DischargeCountry']).sum().T
iran_countries.index = pd.to_datetime(iran_countries.index)
iran_countries_monthly = iran_countries.resample('M').sum()
iran_countries_monthly_kbd = iran_countries_monthly.div(iran_countries_monthly.index.daysinmonth.values, axis=0)





japan_customers = iran_exports.loc[('ASIA','Japan'),:].groupby('CurrentOwner').sum()

japan_customers.T.describe()
japan_customers['sum'] = japan_customers[japan_customers.columns.values].sum(axis=1)
#iran_p.index.get_level_values(1)


"""generate a list of tuples to pass into larger dataframe"""
iranian_buyers = [(region,country,buyer) for region, country, buyer, grade in set(iran_exports.index.tolist())]

"""this creates an empty dataframe for use in the loop. We then go through the list we know of iranian buyers and use that to slice the total clipper data and
and then append to the existing. This creates a matrix, with time in days as columns and looking only at the customers we are concerned with"""
#i = iranian_buyers[0]
existing = pd.DataFrame()
for i in iranian_buyers:
    filtered_list = pd.concat([flows.loc[i,:]], keys=[i])
    existing = pd.concat([existing, filtered_list])

""" figure out the average sulphur countent for the grades and the unique Iranian grades"""
sulphur_dict = flows_data['Sulphur'].groupby([flows_data['CleanGrade']]).mean().to_dict()
iranian_grades = flows_data[flows_data['LoadCountry'] == 'Iran']['CleanGrade'].unique()

"""create a dictionary with only the grades we are concerned with just to double check sulphur content"""
grades_dict = {}
for i in iranian_grades:
    grades_dict[i] = sulphur_dict[i]
    
""" see which grades in our larger dictionary are close to the conditions""" 
substitutes = {}
for i in sulphur_dict.items():
    if i[1] > 1.0:
        substitutes[i[0]] = i[1]
        
"""with this list of substitutes, we can filter our main list based on who has taken and the possible crudes of substitution we are concerned about"""
subs = list(substitutes.keys())
iran_subs_sour = existing.loc[(slice(None),slice(None),slice(None),subs),:]    
iran_subs_sour = iran_subs_sour.T
iran_subs_sour.index = pd.to_datetime(iran_subs_sour.index)

"""weekly totals based on load volumes"""
iran_subs_sour = iran_subs_sour.resample('W', how='sum')
iran_subs_sour = iran_subs_sour.T


iran_subs_sour.resample('M').sum()



iran_subs_sour
iran_subs_sour.index.get_level_values(2)


reindexed = iran_p.set_index(['DischargeRegion','DischargeCountry','CurrentOwner','CleanGrade'])





test = iran_p.loc[:,('ASIA')]
test.columns
iran_p.head()

"""groupby on multi index"""
mike = iran_p.groupby(level=[0,1,2], axis=1).mean()
mike.columns
print(mike)

for region, country, owner in iran_p

iran_p.index

dfOrigPos.loc[(dfOrigPos['STRATEGY'] == )]
