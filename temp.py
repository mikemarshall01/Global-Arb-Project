# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta 

ref_data = pd.read_excel('M://test_data_for_ref_resample.xlsx', sheetname=0, header=0)

start_date = ref_data.iloc[0,4]
end_date = ref_data.iloc[0,5]
cap_offline = ref_data.iloc[0,6]

num_days_offline = end_date - start_date


