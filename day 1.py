#!/usr/bin/env python
# coding: utf-8

# #### Tesla Stock Price Financial Analysis

# #### Data Analysis day 1 using pandas and matplotlib

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("Display.max_column",None)
pd.set_option("Display.max_rows",None)


# In[2]:


#import the data
tsl = pd.read_csv(r"C:\Users\jadha\Downloads\TSLA.csv")


# In[3]:


#plot stock data by line Graph
plt.figure(figsize=(14,10))
tsl['Close'].plot()
plt.show()


# In[4]:


df_close = tsl['Close']
plt.figure(figsize=(14,10))
df_close.plot(style='k.')
plt.title('Scatter plot of closing price')
plt.show()


# #### Create new colume of price differnce

# In[5]:


tsl['price_diff']=tsl['Close'].shift(-1)-tsl['Close']
tsl['price_diff']


# #### Create new colume for daily return

# In[6]:


tsl['daily_rn']=tsl['price_diff']/tsl['Close']
tsl['daily_rn']


# ###### Here we apply rolling widow calculation for 50 days. 
# '''In statistics, a moving average (rolling average or running average) is a calculation to analyze data points by creating a series
#  of averages of different subsets of the full data set. It is also called a moving mean (MM). or rolling mean and is a type of 
# finite impulse response filter. Variations include: simple, cumulative, or weighted forms (described below). Highly used for financial analyses.'''

# In[7]:


tsl['ra50']=tsl['Close'].rolling(50).mean()
plt.figure(figsize=(14,10))
tsl['ra50'].plot()
tsl['Close'].plot()
plt.show()


# 1) Here we have ten year chart of tesla stock with simple moveing average  the blue line is shorter 50 days moveing average most trader will use the cross of short term moving averagae.
# 
# 2) To initial a moveing average to inital a long position and identify  the start of bullish trend.

# In[8]:


tsl['ra50']=tsl['Close'].rolling(50).mean()
tsl['ra10']=tsl['Close'].rolling(10).mean()


# In[9]:


tsl['ra50']


# In[18]:


tsl=tsl.dropna()


# In[11]:


tsl['Shares']=[1 if tsl.loc[ei,'ra10']>tsl.loc[ei,'ra50']else 0 for ei in tsl.index]


# In[12]:


# Calculate Profit


# In[13]:


tsl['Close1']=tsl['Close'].shift(-1)
tsl['profit']=[tsl.loc[ei,'Close1']-tsl.loc[ei,'Close']if tsl.loc[ei,'Shares']==1 else 0 for ei in tsl.index]
tsl['profit'].plot()
plt.axhline(y=0,color='red')


# In[14]:


tsl['log_return']=np.log(tsl['Close'].shift(-1))-np.log(tsl['Close'])
tsl['log_return']


# In[15]:


from scipy.stats import norm
mu = tsl['log_return'].mean()
sigma = tsl['log_return'].std(ddof=1)

density = pd.DataFrame()
density['x'] = np.arange(tsl['log_return'].min()-0.01, tsl['log_return'].max()+0.01, 0.001)
density['pdf'] = norm.pdf(density['x'], mu, sigma)

tsl['log_return'].hist(bins=50, figsize=(15, 8))
plt.plot(density['x'], density['pdf'], color='red')
plt.show()


# In[16]:


prob_return1 = norm.cdf(-0.10, mu, sigma)
print('The probability of dropping over 10% in one day ', prob_return1)


# In[17]:


mu220 = 365*mu
sigma220 = (365**0.5) * sigma
drop20 = None
print('The probability of dropping over 25% over a year: ', drop20)


# In[ ]:




