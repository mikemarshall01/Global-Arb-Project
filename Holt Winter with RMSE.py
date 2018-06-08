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
import datetime as dt
import numpy as np

clipper_db_connection = pypyodbc.connect('Driver=SQL Server Native Client 11.0;'
                                'Server=STCHGS126;'
                                'Database=STG_Targo;'
                                'uid=mima;'
                                'Trusted_Connection=Yes;')

clipper_db_query = '''exec sp_CurrentStems @datasource = 'clipper' '''
clipper_db_table = pd.read_sql(clipper_db_query, clipper_db_connection)



### get load dat6e column in same format as decades from balances
clipper_db_table['days'] = pd.to_datetime(clipper_db_table['loaddate']).dt.day
clipper_db_table['months'] = pd.to_datetime(clipper_db_table['loaddate']).dt.month
clipper_db_table['years'] = pd.to_datetime(clipper_db_table['loaddate']).dt.year
clipper_db_table['days'] = np.where(clipper_db_table.days <= 10, 1, np.where((10<clipper_db_table.days)&(clipper_db_table.days <=20), 11, 21)).astype('int64')
clipper_db_table['LD'] = pd.to_datetime(clipper_db_table[['days','months','years']])

#create pivot table for to get input data
clipper_db_pivot = pd.pivot_table(clipper_db_table, values='quantity', index = clipper_db_table['LD'], columns = ['loadcountry','dischargesubregion'], aggfunc = 'sum' )
clipper_db_pivot.head()




refdb_pivot = pd.pivot_table(refdb_table, values = 'amount', index='period', columns = ['unittype', 'country'], aggfunc='sum')
NWE_Total_ref = refdb_pivot.groupby(level='unittype', axis=1).sum()
NWE_Total_ref.head()


saud_data = pd.read_excel("M://saudi data.xlsx").transpose()
saud_data['DaysInMonth'] = pd.to_datetime(saud_data.index)
saud_data['DaysInMonth'] = saud_data['DaysInMonth'].dt.daysinmonth
kbd = saud_data.div(saud_data['DaysInMonth']*1000, axis = 'index').fillna(0).drop(['DaysInMonth'], axis = 1)


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
		boundaries = [(0, 1), (0, 1), (0, 1)]
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


# =============================================================================
# data = pd.read_excel("M:\saud exports for holt test.xlsx")
# saud_exports = list(data.iloc[:,1].values)
# saud_exports[:]
# additive(list(kbd['CHINA']), 12,12)
# multiplicative(list(kbd['INDIA EC']), 12,12)
# =============================================================================

forecast_group = pd.DataFrame()

for i in kbd.columns.values:
    forecast_group = pd.concat([forecast_group, pd.DataFrame(additive(list(kbd[i]), 12,12)[0]).rename(columns={0:i})], axis=1)

path='M:/'
forecast_group.to_excel(path+'AutoForecast.xlsx')
    

    #multiplicative(list(kbd['INDIA EC']), 12,12
    

