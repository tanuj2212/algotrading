#!/usr/bin/env python
# coding: utf-8

# # Algorithmic trading with Keras (using LSTM)
# 
# We use a Long Short Time Memory recurrent neural network to develop a good trading strategy for the S&P 500 index: the first trading day of each month we want our model to tell us if we are going to stay in the market for the current month or not.
# 
# We verify that, in a period of 4 years which comprehends the 2008 crisis, this LSTM-trading-strategy is far better than the buy and hold strategy (stay always in the market) and the moving average strategy (buy when the current price is greater or equal to the moving average of past 12 months and sell otherwise). 
# 
# We compute the gross and net yield (as it is by the Italian law: 26% tax on capital gain and 0.10% fee to the broker at each transaction): our model performed roughly a 10% net annual yield (which is not bad, considering the 2008 crisis)

# # first we import all required libraries 

# In[1]:


from __future__ import division
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Conv1D,Flatten,MaxPooling1D,LSTM
from keras.callbacks import EarlyStopping,ModelCheckpoint,TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


# ## Part 1: Get the data 
# 
# I used Yahoo Finance to get the data of the S&P 500 index from 1973/1/1 to 2011/3/31. Our analysis is monthly-based, and all the decisions are made the first trading day of the month.
# For a reason which will be clarified by the following code, our analysis will start from 24 months after January 1973 and end the month before March 2011.
# 
# Since it is not possible to use Yahoo Finance on this kernel, I saved the data in "GSPC.csv". I still left the original code commented.
# 

# In[2]:


import yfinance as yf
import plotly.graph_objs as go
import pandas_datareader as pdr


# In[3]:


start_date=datetime.datetime(1980, 12, 12)
end_date=datetime.date.today()


# In[4]:


df = pdr.get_data_yahoo('AAPL', start=start_date, end=end_date)
df.drop("Adj Close",axis=1,inplace=True)
df.to_csv("AAPL.csv")


# In[5]:


df.head()


# # Historical Data Visualization

# In[6]:


plt.figure(figsize=(25,7))
plt.title('historical price')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close_Price', fontsize=18)
plt.grid()
plt.show()


# In order to develop our trading strategy, we need to obtain a dataframe with the monthly means and the first trading day of the month price. First we obtain the monthly means

# In[7]:


dfm=df.resample("M").mean()

dfm=dfm[:-1] # As we said, we do not consider the month of end_date

dfm.head(15)


# Then we obtain the list of the first trading day for each month

# In[8]:


start_year=start_date.year
start_month=start_date.month
end_year=end_date.year
end_month=end_date.month

first_days=[]
# First year
for month in range(start_month,13):
    first_days.append(min(df[str(start_year)+"-"+str(month)].index))
# Other years
for year in range(start_year+1,end_year):
    for month in range(1,13):
        first_days.append(min(df[str(year)+"-"+str(month)].index))
# Last year
for month in range(1,end_month+1):
    first_days.append(min(df[str(end_year)+"-"+str(month)].index))


# In[9]:


first_days


# Now for each month we have the means of the month, the first trading day of the current month (and its open price) and the first trading day of the next month (and its open price).
# 
# The feature *rapp* is the quotient between the open price of the first trading day of the next month and the open price of the first trading day of the current month. It will be used because it gives the variation of the portfolio for the current month

# In[10]:


dfm["fd_cm"]=first_days[:-1] #current month 
dfm["fd_nm"]=first_days[1:] #next month
dfm["fd_cm_open"]=np.array(df.loc[first_days[:-1],"Open"]) #current month open price
dfm["fd_nm_open"]=np.array(df.loc[first_days[1:],"Open"]) #next month open price
dfm["rapp"]=dfm["fd_nm_open"].divide(dfm["fd_cm_open"]) #


# In[11]:


dfm.head()


# In[12]:


dfm.tail()


# Now we add the columns corresponding to the moving averages at 1 and 2 years

# In[13]:


dfm["mv_avg_12"]= dfm["Open"].rolling(window=12).mean().shift(1)
dfm["mv_avg_24"]= dfm["Open"].rolling(window=24).mean().shift(1)


# In[14]:


dfm.head(13)


# Note that in this way each month of dfm contains the moving averages of the previous 12 and 24 months (excluding the current month)

# In[15]:


print(dfm.loc["1980-12","mv_avg_12"])
print(dfm.loc["1980-12":"1981-11","Open"])
print(dfm.loc["1980-12":"1981-11","Open"].mean())


# here we can see that mean of last 12 months is equal to the my_avg_12 of 13th month

# In[16]:


dfm=dfm.iloc[24:,:] # WARNING: DO IT JUST ONE TIME!
dfm.index


# Finally, we can divide *dfm* in train and test set

# In[17]:


dfm


# In[18]:


mtest=100
train=dfm.iloc[:-mtest,:] 
test=dfm.iloc[-mtest:,:] 


# In[19]:


train.shape


# In[20]:


test.shape


# ## Part 2: Define functions to compute gross and net yield
# 
# Notice that the gross yield can be computed very easily using the feature *rapp*.
# The following function explains how: the vector v selects which months we are going to stay in the market

# In[21]:


# This function returns the total percentage gross yield and the annual percentage gross yield
#
def yield_gross(df,v):
    prod=(v*df["rapp"]+1-v).prod()
    n_years=len(v)/12
    return (prod-1)*100,((prod**(1/n_years))-1)*100


# We just need to define a function to compute the net yield, considering a 15% tax on capital gain and 0.10% commission to the broker at each transaction

# here we consider the short term capital gain so tax on capital gain is 15%

# In[22]:


tax_cg=0.15
comm_bk=0.001


# In[23]:


# This function will be used in the function yield_net

# Given any vector v of ones and zeros, this function gives the corresponding vectors of "islands" of ones of v
# and their number. 
# For example, given v = [0,1,1,0,1,0,1], expand_islands2D gives
# out2D = [[0,1,1,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,0,1]] and N=3

def expand_islands2D(v):
    
    # Get start, stop of 1s islands
    v1 = np.r_[0,v,0]
    idx = np.flatnonzero(v1[:-1] != v1[1:])
    s0,s1 = idx[::2],idx[1::2]
    if len(s0)==0:
        return np.zeros(len(v)),0
    
    # Initialize 1D id array  of size same as expected o/p and has 
    # starts and stops assigned as 1s and -1s, so that a final cumsum
    # gives us the desired o/p
    N,M = len(s0),len(v)
    out = np.zeros(N*M,dtype=int)

    # Setup starts with 1s
    r = np.arange(N)*M
    out[s0+r] = 1


    # Setup stops with -1s
    if s1[-1] == M:
        out[s1[:-1]+r[:-1]] = -1
    else:
        out[s1+r] -= 1

    # Final cumsum on ID array
    out2D = out.cumsum().reshape(N,-1)
    return out2D,N


# Again, the vector v selects which months we are going to stay in the market

# In[24]:


# This function returns the total percentage net yield and the annual percentage net yield

def yield_net(df,v):
    n_years=len(v)/12
    
    w,n=expand_islands2D(v)
    A=(w*np.array(df["rapp"])+(1-w)).prod(axis=1)  # A is the product of each island of ones of 1 for df["rapp"]
    A1p=np.maximum(0,np.sign(A-1)) # vector of ones where the corresponding element if  A  is > 1, other are 0
    Ap=A*A1p # vector of elements of A > 1, other are 0
    Am=A-Ap # vector of elements of A <= 1, other are 0
    An=Am+(Ap-A1p)*(1-tax_cg)+A1p
    prod=An.prod()*((1-comm_bk)**(2*n)) 
    
    return (prod-1)*100,((prod**(1/n_years))-1)*100   


# ## Part 3: Define the LSTM model

# We want to use a LSTM neural network to decide, the first day of each day of the test period, whether we are going to stay in the market for the month or not.
# 
# We reshape the data (the LSTM wants the data in a particular shape, involving "windows") and at each step we want to predict the opening price of the first day of the next month: in this way we will be able to find the vector v which selects the months during which we are going to stay in the market

# In[25]:


def create_window(data, window_size = 4):    
    data_s = data.copy()
    for i in range(window_size):
        data = pd.concat([data, data_s.shift(-(i + 1))], axis = 1)
        
    data.dropna(axis=0, inplace=True)
    return(data)


# In[26]:


create_window(dfm)


# In[27]:


a= np.array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
np.reshape(a,(3,1,5))


# In[28]:


a= np.array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
np.reshape(a,(3,5,1))


# In[29]:


scaler=MinMaxScaler(feature_range=(0,1))
dg=pd.DataFrame(scaler.fit_transform(dfm[["High","Low","Open","Close","Volume","fd_cm_open",                                          "mv_avg_12","mv_avg_24","fd_nm_open"]].values)) # fit transform is used for scale the data
dg0=dg[[0,1,2,3,4,5,6,7]]


window=4
dfw=create_window(dg0,window)

X_dfw=np.reshape(dfw.values,(dfw.shape[0],window+1,8))
print(X_dfw.shape)
print(dfw.iloc[:4,:])
print(X_dfw[0,:,:])

y_dfw=np.array(dg[8][window:])


# In[30]:


dg0


# In[31]:


X_trainw=X_dfw[:-mtest-1,:,:]
X_testw=X_dfw[-mtest-1:,:,:]
y_trainw=y_dfw[:-mtest-1]
y_testw=y_dfw[-mtest-1:]


# In[32]:


X_trainw.shape


# In[33]:


X_testw.shape


# In[34]:


y_trainw.shape


# In[35]:


y_testw.shape


# In[36]:


def model_lstm(window,features):
    
    model=Sequential()
    model.add(LSTM(300, input_shape = (window,features), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(200, input_shape=(window,features), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(100,kernel_initializer='uniform',activation='relu'))        
    model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
    model.compile(loss='mse',optimizer='adam')
    
    
    return model


# In[37]:


model=model_lstm(window+1,8)
history=model.fit(X_trainw,y_trainw,epochs=500, batch_size=24,verbose=0)


# In[38]:


plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')
plt.show()


# In[39]:


y_pr=model.predict(X_trainw)


# In[40]:


plt.figure(figsize=(30,10))
plt.plot(y_trainw, label="actual")
plt.plot(y_pr, label="prediction")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Actual open price and pedicted one on train set",fontsize=25)
plt.show()


# In[41]:


y_pred=model.predict(X_testw)


# **We stay in the market when the predicted price for the next month is greater than the current price and stay out otherwise. The vector v indicates the "in months" (as 1s) and "out months" (as 0s)**

# In[42]:


v=np.diff(y_pred.reshape(y_pred.shape[0]),1)
v_lstm=np.maximum(np.sign(v),0)


# In[43]:


plt.figure(figsize=(30,10))
plt.plot(y_testw, label="actual")
plt.plot(y_pred, label="prediction")
plt.plot(v_lstm,label="In and out")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Actual open price, predicted one and vector v_lstm",fontsize=25)
plt.show()


# **The preceeding plot shows an interesting feature of the prediction of our model: it is quite good at predicting the sign of the first derivative of the index, but this is exactly what we need for our trading strategy!** 

# ## Part 4: Compare the LSTM method with other methods

# Now we can copare our LSTM-trading-strategy with the buy and hold strategy and the moving average strategy. In order to do so we compute the corresponding vectors v_bh and v_ma which select the months during which we are going to stay in the market.

# In[44]:


v_bh=np.ones(test.shape[0])
v_ma=test["fd_cm_open"]>test["mv_avg_12"]


# In[45]:


def gross_portfolio(df,w):
    portfolio=[ (w*df["rapp"]+(1-w))[:i].prod() for i in range(len(w))]
    return portfolio


# In[46]:


plt.figure(figsize=(30,10))
plt.plot(gross_portfolio(test,v_bh),label="Portfolio Buy and Hold")
plt.plot(gross_portfolio(test,v_ma),label="Portfolio Moving Average")
plt.plot(gross_portfolio(test,v_lstm),label="Portfolio LSTM")
plt.legend(fontsize=20)
plt.grid(axis="both")
plt.title("Gross portfolios of three methods", fontsize=25)
plt.show()


# In[47]:


print("Test period of {:.2f} years, from {} to {} \n".format(len(v_bh)/12,str(test.loc[test.index[0],"fd_cm"])[:10],      str(test.loc[test.index[-1],"fd_nm"])[:10]))

results0=pd.DataFrame({})
results1=pd.DataFrame({})
results2=pd.DataFrame({})
results3=pd.DataFrame({})

results0["Method"]=["Buy and hold","Moving average","LSTM"]
results1["Method"]=["Buy and hold","Moving average","LSTM"]
results2["Method"]=["Buy and hold","Moving average","LSTM"]
results3["Method"]=["Buy and hold","Moving average","LSTM"]

vs=[v_bh,v_ma,v_lstm]
results0["Total gross yield"]=[str(round(yield_gross(test,vi)[0],2))+" %" for vi in vs]
results1["Annual gross yield"]=[str(round(yield_gross(test,vi)[1],2))+" %" for vi in vs]
results2["Total net yield"]=[str(round(yield_net(test,vi)[0],2))+" %" for vi in vs]
results3["Annual net yield"]=[str(round(yield_net(test,vi)[1],2))+" %" for vi in vs]

print(results0)
print("\n")
print(results1)
print("\n")
print(results2)
print("\n")
print(results3)


# In[ ]:





# In[ ]:




