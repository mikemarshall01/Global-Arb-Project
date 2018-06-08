# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:29:34 2018

@author: mima
"""

import pandas as pd
df = pd.DataFrame({
    'a': [1,2,3,4,5],
    'b': [5,4,3,3,4],
    'c': [3,2,4,3,10],
    'd': [3, 2, 1, 1, 1]
})

params = {'a': 2.5, 'b': 3.0, 'c': 1.3, 'd': 0.9}

df1 = df.assign(**params).mul(df).sum(1)
print (df1)


d = [{'type_id': 6, 'type_name': 'Type 1'}, {'type_id': 12, 'type_name': 'Type 2'}]
print([{'type':x['type_id'],'name':x['type_name']} for x in d])

[{'type':x['type_id'],'name':x['type_name']} for x in d]

d[1]

{'type':d[1]['type_id']}

def add(x,y):
    z = 4
    def add_z():
        return z+2
    return x+y+z

add(2,3)

