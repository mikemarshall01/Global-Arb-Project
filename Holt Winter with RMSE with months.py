# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 14:31:13 2018

@author: mima
"""
#from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

from __future__ import division
from sys import exit
from math import sqrt
from numpy import array
from scipy.optimize import fmin_l_bfgs_b
import pandas as pd
import pypyodbc
from datetime import datetime as dt
from datetime import date
from datetime import timedelta
import numpy as np



# =============================================================================
# must define what our loss function is for the optimiser - here we are using the root mean squared error
# 1) Y is the data series we are looking at
# 2) the type is defined as 'linear', 'mulitiplicative' or 'additive' - will be using additive but others in here for good measure
# and takes the second positional argument
# 3) set rmse = 0 initially
# 4) a is the initial level
# 5) b is the initial trend
# 6) y is the expected value of Y which takes the inital values of a plus the initial values of b, i.e. first level with trend added
# 7) this will give a series of y which we then measure against the true value Y and calculate the rmse
# 8) s is the seaosnal index which when rearranged gives s equation below when you bound inital coefficient to be between 1 and 0
#  https://www.otexts.org/fpp/7/5
# 9) sum of the inidividual components are then added to list y
# 10) the arguments that are passed into the RMSE are given in the optimiser algo
# =============================================================================

def RMSE(params, *args):

	Y = args[0]
	type = args[1]
	rmse = 0

	if type == 'linear':

		alpha, beta = params
		a = [Y[0]] # first data point
		b = [Y[1] - Y[0]] # trend is first minus second, or change in level
		y = [a[0] + b[0]] # hence estimate is first value plus first trend estimate

		for i in range(len(Y)):
            # for each sequential point in our data series, update the level and trend calcs and add to estimate list
			a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
			b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
			y.append(a[i + 1] + b[i + 1])

	else:

		alpha, beta, gamma = params
		m = args[2]		
		a = [sum(Y[0:m]) / float(m)]
		b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]

		if type == 'additive':

			s = [Y[i] - a[0] for i in range(m)] # initial seasonal indicies
			y = [a[0] + b[0] + s[0]] # initial observation is level plus trend plus seasonality

			for i in range(len(Y)):

				a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i])) # take actual minus seasonbal index weighed agfainst expected 
				b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i]) # current slope estimate based on levels estimate weighed against previous trend value
				s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i]) # seasonality is rearranged version same as above but sub in the levels equation above and bound results in this
				y.append(a[i + 1] + b[i + 1] + s[i + 1])

		elif type == 'multiplicative':

			s = [Y[i] / a[0] for i in range(m)]
			y = [(a[0] + b[0]) * s[0]]

			for i in range(len(Y)):

				a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
				b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
				s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
				y.append((a[i + 1] + b[i + 1]) * s[i + 1])

		else:

			exit('Type must be either linear, additive or multiplicative')
		
	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))

	return rmse

def linear(x, fc, alpha = None, beta = None):

	Y = x[:]

	if (alpha == None or beta == None):

		initial_values = array([0.3, 0.1])
		boundaries = [(0, 1), (0, 1)]
		type = 'linear'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type), bounds = boundaries, approx_grad = True)
		alpha, beta = parameters[0]

	a = [Y[0]]
	b = [Y[1] - Y[0]]
	y = [a[0] + b[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append(a[-1] + b[-1])

		a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
		b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
		y.append(a[i + 1] + b[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, rmse

def additive(x, m, fc, alpha = None, beta = None, gamma = None):

	Y = x[:]

	if (alpha == None or beta == None or gamma == None):

		initial_values = array([0.3, 0.1, 0.1])
		boundaries = [(0, 0.5), (0, 0.5), (0, 0.5)]
		type = 'additive'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
		alpha, beta, gamma = parameters[0]

	a = [sum(Y[0:m]) / float(m)]
	b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
	s = [Y[i] - a[0] for i in range(m)]
	y = [a[0] + b[0] + s[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append(max((a[-1] + b[-1] + s[-m]),0))

		a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
		b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
		s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
		y.append(a[i + 1] + b[i + 1] + s[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, gamma, rmse

def multiplicative(x, m, fc, alpha = None, beta = None, gamma = None):

	Y = x[:]

	if (alpha == None or beta == None or gamma == None):

		initial_values = array([0.0, 1.0, 0.0])
		boundaries = [(0, 1), (0, 1), (0, 1)]
		type = 'multiplicative'

		parameters = fmin_l_bfgs_b(RMSE, x0 = initial_values, args = (Y, type, m), bounds = boundaries, approx_grad = True)
		alpha, beta, gamma = parameters[0]

	a = [sum(Y[0:m]) / float(m)]
	b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
	s = [Y[i] / a[0] for i in range(m)]
	y = [(a[0] + b[0]) * s[0]]
	rmse = 0

	for i in range(len(Y) + fc):

		if i == len(Y):
			Y.append(max(((a[-1] + b[-1]) * s[-m]),0))

		a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
		b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
		s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
		y.append((a[i + 1] + b[i + 1]) * s[i + 1])

	rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:-fc], y[:-fc - 1])]) / len(Y[:-fc]))

	return Y[-fc:], alpha, beta, gamma, rmse

clipper_db_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STCHGS126;'
                                'Database=STG_Targo;'
                                'uid=mima;'
                                'Trusted_Connection=Yes;')

clipper_db_query = '''exec sp_CurrentStems @datasource = 'clipper' '''
clipper_db_table = pd.read_sql(clipper_db_query, clipper_db_connection)

# This is the adjusted table from the US balances

clipper_db_table = pd.read_excel('M:\CLIPPER_TARGO.xlsx', sheet_name = 'CLIPPER_TARGO')

### get load dat6e column in same format as decades from balances
clipper_db_table['months'] = pd.to_datetime(clipper_db_table['loaddate']).dt.month
clipper_db_table['years'] = pd.to_datetime(clipper_db_table['loaddate']).dt.year
clipper_db_table['days'] = 1
clipper_db_table['LD'] = pd.to_datetime(clipper_db_table[['days','months','years']]).dt.date

#create pivot table for to get input data
clipper_db_pivot = pd.pivot_table(clipper_db_table, values='quantity',
                                  index = clipper_db_table['LD'],
                                  columns = ['loadcountry','dischargesubregion'],
                                  aggfunc = 'sum' ).fillna(0)

# flatten the table with the new combined column headers and export
clipper_db_pivot.columns = list([clipper_db_pivot.columns[x][0]+'_'+clipper_db_pivot.columns[x][1] for x in range(0,len(clipper_db_pivot.columns))])

# slice so we only use complete data, i.e. form 2013 onwards
clipper_db_pivot = clipper_db_pivot.iloc[(clipper_db_pivot.index > date(2014,12,31))&
                                         (clipper_db_pivot.index < (pd.to_datetime('today').date()-timedelta(days=30)))]

# transform to kbd
clipper_db_pivot['DaysInMonth'] = pd.to_datetime(clipper_db_pivot.index)
clipper_db_pivot['DaysInMonth'] = clipper_db_pivot['DaysInMonth'].dt.daysinmonth
clipper_db_pivot = clipper_db_pivot.div(clipper_db_pivot['DaysInMonth']*1000, axis = 'index').fillna(0).drop(['DaysInMonth'], axis = 1)

# create an index for the forecast dates: we want to have a date range from current month onwards, data completeness
fc_index = pd.DataFrame(pd.date_range(pd.to_datetime('today').date()-timedelta(days=30), periods=12, freq='MS'))
fc_index.apply(lambda x: x.dt.strftime('%Y/%m/%d'))
#fc_index = pd.to_datetime(fc_index)
fc_index.loc[12] = str('RMSE')
fc_index.loc[13] = str('RMSE vs Mean')

    
for i in clipper_db_pivot.columns.values:
    model_output = additive(list(clipper_db_pivot[i]), 12,12)
    temp = list(model_output[0])
    temp.append(model_output[4])
    temp.append(model_output[4] / np.mean(model_output[0]))
    temp = pd.DataFrame(temp).rename(columns={0:i})
    fc_index = pd.concat([fc_index, temp], axis=1)    
    
# set the index of the forecast dataframe to the 0th column we created above  
fc_index = fc_index.set_index(0)    

# export as an excel file
path='M:/'
fc_index.to_excel(path+'FlowsForecasts.xlsx')

# Seaonality charts for visualisations
import seaborn as sns
import matplotlib.pyplot as plt
visualisation = pd.concat([clipper_db_pivot, fc_index], axis=0)

# Slice the dataframe to cut off the RMSE and the RMSE:mean, convert datetime index    
data = pd.DataFrame(visualisation.iloc[:-3,:])
data.index = pd.to_datetime(data.index)

# What do we want to look at?
load_country = 'United States'
dest_region = 'CHINA'

#for dest_region in regwions

x = "{}_{}".format(load_country, dest_region)
seasonal_grid = pd.pivot_table(pd.DataFrame(data[x]), index=data.index.month, columns=data.index.year, aggfunc='sum')
sns.set()
seasonal_grid.plot()
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

seasonal_grid.columns.values

#####

# =============================================================================
# =IFERROR(INDEX([FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$13,MATCH((EOMONTH(T$15,-1)+1), [FlowsForecasts.xlsx]Sheet1!$A$1:$A$13,0),MATCH($A83&"_"&VLOOKUP($B83, REF!$A$2:$F$50,2,0),[FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$1,0)),0)
# +IFERROR(INDEX([FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$13,MATCH((EOMONTH(T$15,-1)+1), [FlowsForecasts.xlsx]Sheet1!$A$1:$A$13,0),MATCH($A83&"_"&VLOOKUP($B83, REF!$A$2:$F$50,3,0),[FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$1,0)),0)
# +IFERROR(INDEX([FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$13,MATCH((EOMONTH(T$15,-1)+1), [FlowsForecasts.xlsx]Sheet1!$A$1:$A$13,0),MATCH($A83&"_"&VLOOKUP($B83, REF!$A$2:$F$50,4,0),[FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$1,0)),0)
# +IFERROR(INDEX([FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$13,MATCH((EOMONTH(T$15,-1)+1), [FlowsForecasts.xlsx]Sheet1!$A$1:$A$13,0),MATCH($A83&"_"&VLOOKUP($B83, REF!$A$2:$F$50,5,0),[FlowsForecasts.xlsx]Sheet1!$A$1:$AOA$1,0)),0)
# for i in clipper_db_pivot.columns.values:
#    forecast_group = pd.concat([forecast_group, pd.DataFrame(additive(list(clipper_db_pivot[i]), 12,12)[0]).rename(columns={0:i})], axis=1)
# =============================================================================
