# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:02:13 2018

@author: mima
"""

#api to BT conversion

'''
Add 131.5 to the API gravity. The formula for API gravity is API = (141.5/SG)
-131.5 where SG is the specific gravity of the petroleum liquid being
measured. For example, for an API gravity of 50, add 131.5 to obtain 181.5.

Divide 141.5 by (131.5 + API gravity) to obtain the specific gravity of the
oil. Continuing the example, divide 141.5 by 181.5 from the last step to get
.7796.

Multiply the specific gravity of the oil by the density of water to obtain the
density of the oil. This follows from the formula of specific gravity where
SG = density(oil)/density(water). Concluding the example, .7796 * 1 grams per
cubic centimeter = .7796 g/cc.

'''

import pandas as pd
import pypyodbc

cxn = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STUKLS022;'
                                'Database=TradeTracker;'
                                'uid=gami;'
                                'Trusted_Connection=Yes;')

query = """
        SELECT Assay.AssayID, Assay.Grade, Assay.API, Assay.Density
        FROM TradeTracker.dbo.Assay Assay
        """
        
crudes = pd.read_sql(query, cxn)

def conversion_to_BT(crudes):
    return 1 / ((141.5 / (131.5 + crudes['api'])) * 0.159)    
        
crudes['bt'] = crudes.apply(conversion_to_BT, axis=1)
cxn.close()


path='L:/TRADING/ANALYSIS/Python/MIMA/'
crudes.to_excel(path+'grade_densities.xlsx')
