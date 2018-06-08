# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:14:16 2018

@author: mima
"""
'''
website: https://grisha.org/blog/2016/02/17/triple-exponential-smoothing-forecasting-part-iii/
'''

series = [3,10,12,13,12,10,12]
series[-3]

def average(series, n=None):
    if n is None:
        return average(series,len(series))
    return float(sum(series[-n:]))/n

def weighted_average(series, weights):
    result = 0.0
    weights.reverse()
    for n in range(len(weights)):
        result += series[-n-1] * weights[n]
    return result

weights = [0.1,0.2,0.3,0.4]

weighted_average(series, weights)

# Given a series and alpha, return series of smoothed points:
def exponential_smoothing(series, alpha):
    result = [series[0]] # first value is the same as the series
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1-alpha) * result[n-1])
    return result

exponential_smoothing(series, 0.1)

''' 
first observed trend is first expected. I.E. we take the change between first
and second values band apply that as our trend expected value
'''

# given a series and alpha, return smoothed points
def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series): #we are forecasting
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result


double_exponential_smoothing(series, 0.9, 0.9)


series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
          27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
          26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
          18,8,17,21,31,34,44,38,31,30,26,32]





# first we need to see what our initial trend is. This is average of all the seaonaed trends over the length of the season (year). i.e. Jan to jan, feb to feb, etc
def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen


# Then we need to determine what the initial seaonal inidicies are 
def initial_seasonal_components(series,slen):
    seasonals = {} # initialise an empty dictionary to hold the values
    season_averages = [] # initialise an empty list to hold the final averages
    n_seasons = int(len(series)/slen) # number of seasons is the length of the data series divided by season length
    
    # compute season averages
    for j in range (n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen)) # calculate the averages for each month in year j, 
                                                                            # so we have an array of length slen appended to season_averages
    # compute initial values
    for i in range(slen): # for which period within the season, in our case, month
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons): # for which season or year
            # this is saying for i (the month we are concerned about), find how much the value of i differed from the yearly average. 
            # For example, when j = 0, then this is the first year and when i = 0 its the first month
            sum_of_vals_over_avg += series[slen*j+i] - season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons # we then assign the sum of the values divided by the number of seaons in the data 
        #for that month across the years to the seaosnals dictionary 
    return seasonals

### The algo makes use of the two functions above

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # inital values
            smooth = series[0]
            trend = inital_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) +1 # how far in the future are we looking
            result.append((smooth + m*trend) + seasonals[i%slen]) # i%slen essentially repeats a repeating pattern in incremental amounts up to the values of slen, 
                                                                    # so in this case, we want the seasonals to be jan to dec over and over again, so we have i % sln to get 0-11 
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth - last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[ i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


####### - need to add solver based on minimising the SSE of above
    






slen = 12
j = 2
series[slen*j:slen*j+slen]    

initial_seasonal_components(series,12)           
                
            22%6
            
for i in range(20):
    print(i,i%5)
    
for num in range(2, 10):
    if num % 2 == 0:
        print("Found an even number", num)
    else:
        print("Found a number", num)
            