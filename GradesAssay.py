# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 15:31:44 2018

@author: mima
"""

import pandas as pd
import re
import pypyodbc

#GradesToCheck = pd.read_csv("L:\TRADING\ANALYSIS\Python\MIMA\ClipperTargoStuff\GradesNeedSulphur.txt")

cxn = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STCHGS112;'
                                'Database=STG_Targo;'
                                'uid=mima;'
                                'Trusted_Connection=Yes;')

query = '''

        SELECT GradeLocationAssay.[Id]
              ,[IdRegionTree]
              ,Grade.Name
              ,[IdLocation]
              ,[IdAssay]
              ,[NumValue]
              ,[StartDate]
        FROM [STG_Targo].[dbo].[GradeLocationAssay]
          INNER JOIN [STG_Targo].[dbo].[Grade] as Grade on [STG_Targo].[dbo].[GradeLocationAssay].IdGrade = Grade.Id
        WHERE [STG_Targo].[dbo].[GradeLocationAssay].idassay not in (49)
        ORDER BY [Name]
  
'''
GradesWithSulphur = pd.read_sql(query, cxn)
GradesWithSulphur = pd.Series(GradesWithSulphur.name.unique()).str.upper().tolist()
GradesToCheck = GradesToCheck.CleanGrade.str.upper().tolist()

matches = []

for grade in GradesToCheck:
    pattern = re.compile(grade[:4])
    temp_list = [x for x in GradesWithSulphur if pattern.match(x)]
    matches.append(temp_list)

checking = list(zip(GradesToCheck, matches))
ListForExcel = list([x[1] for x in checking])

type(GradesToCheck.CleanGrade)