# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:07:34 2018

@author: mima
"""

# =============================================================================
# *** FROM 2005 ****
# =============================================================================

import pypyodbc
import pandas as pd

refdb_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=RefineryInfo;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')

refdb_query = '''
                SELECT RefineryStatusInPeriod.Region, RefineryStatusInPeriod.SubRegion, Country.Country, RefineryStatusInPeriod.RefineryName, RefineryStatusInPeriod.Status, RefineryStatusInPeriod.UnitName, RefineryStatusInPeriod.UnitType, RefineryStatusInPeriod.PeriodType, RefineryStatusInPeriod.Period, RefineryStatusInPeriod.Amount
                FROM RefineryInfo.dbo.RefineryStatusInPeriod RefineryStatusInPeriod
                LEFT JOIN RefineryInfo.dbo.Refinery Refinery
                ON  RefineryStatusInPeriod.RefineryName = Refinery.RefineryName
                INNER JOIN Country Country
                ON Refinery.Country = Country.CountryID
                WHERE (RefineryStatusInPeriod.Region='NW EUROPE')
                AND PeriodType = 'Month'
                AND Status = 'Avail Capacity'
                '''
                
refdb_table = pd.read_sql(refdb_query, refdb_connection) 
refdb_pivot = pd.pivot_table(refdb_table, values = 'amount', index='period', columns = ['unittype', 'country'], aggfunc='sum')
NWE_Total_ref = refdb_pivot.groupby(level='unittype', axis=1).sum()
NWE_Total_ref.head()

mike = pd.Series(refdb_table['unittype'].unique())
