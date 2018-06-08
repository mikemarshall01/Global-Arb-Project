# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 14:31:03 2018

@author: mima
"""

# Match strings in two different lists

import pandas as pd

raw_assay = pd.read_excel('L://TRADING//ANALYSIS//GLOBAL//Arb Models//GPW model RA.xlsm', sheetname = 'Upload_Test', header = 0, index_col = 'Database_Name')

# with the crude name set as the index, we use pd.to_dict to set a dictioanry of dictionaries
crudes = raw_assay.to_dict('index')
crudes.keys()
k = crudes.items()

crudes = []

things = [{'thing_id': row[0], 'thing_name': row[1]} for row in raw_assay]

mike = [print(row[0:5]) for row in raw_assay[0:1].itertuples()]
print(mike)

names=["lloyd", "alice", "tyler"]
keys=["homework", "quizzes", "tests"]
dic={ 
     name.capitalize():
         { 
                 key:'mike' for key in keys
         } for name in names
    }
dic
{'Tyler': {'quizzes': [], 'tests': [], 'homework': []}, 
 'Lloyd': {'quizzes': [], 'tests': [], 'homework': []},
 'Alice': {'quizzes': [], 'tests': [], 'homework': []}}

