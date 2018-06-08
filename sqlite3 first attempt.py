# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:31:22 2017

@author: mima
"""

import sqlite3

conn = sqlite3.connect('tysql.sqlite') 
c = conn.cursor()
c.execute('''
          
CREATE TABLE Products
(
  prod_id    char(10)      NOT NULL ,
  vend_id    char(10)      NOT NULL ,
  prod_name  char(255)     NOT NULL ,
  prod_price decimal(8,2)  NOT NULL ,
  prod_desc  varchar(1000) NULL 
);

''')


