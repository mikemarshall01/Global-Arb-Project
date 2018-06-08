# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:12:20 2018

@author: mima
"""

# =============================================================================
# Journey Times
# =============================================================================

import pypyodbc
import pandas as pd


clipper_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=TradeTracker;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')

clipper_query = '''
                SELECT
                Cargoes."Vessel Name",
                Cargoes."Vessel IMO",
                Cargoes."Grade",
                Cargoes.Volume,
                Cargoes."Cargo Type",
                Cargoes.Shipper,
                Cargoes."Load Date",
                Cargoes."Load Port",
                Cargoes."Discharge Port",
                Cargoes."Clipper Probability",
                Cargoes."Load Asset",
                Cargoes."Discharge Date",
                Cargoes.Refiner,
                Cargoes.API,
                Cargoes.Sulphur,
                Cargoes."LoadMth",
                Cargoes.DischargeMth,
                Cargoes.LoadRegion,
                Cargoes.LoadCountry,
                Cargoes. dischargeRegion,
                Cargoes.DischargeCountry,
                Cargoes.DefaultRefinery,
                Cargoes."SOCAR Default Buyer",
                Cargoes."Default Grade",
                Cargoes.Owner,
                Cargoes.Refinery,
                Cargoes."Vessel Size",
                Cargoes."SOCAR Volume",
                Cargoes.NewVol,
                Cargoes.Export,
                Cargoes.VT,
                Cargoes.LoadDecade,
                Cargoes.DischargeDecade,
                Cargoes.Route,
                Cargoes.LoadSTSVessel,
                Cargoes.OfftakeSTSVessel
                
                
                FROM TradeTracker.dbo.CleanerDataClipper AS Cargoes
                '''
                
clipper_table = pd.read_sql(clipper_query, clipper_connection) 
vt_table = pd.pivot_table(clipper_table, values = 'vt', index='loadregion', columns = 'dischargeregion', aggfunc='mean')
