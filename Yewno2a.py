
# coding: utf-8

# In[68]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[128]:

DataTable = pd.read_csv('Yewno3b')


# In[129]:

DataTable.head()


# In[130]:

Datab2 = DataTable.ix[DataTable['TLT'] < 0]


# In[131]:

Datab2


# In[132]:

X = Datab2[['XLE', 'XLI', 'XLY', 'XLB', 'XLF', 'XLK']]


# In[133]:

y = Datab2['TLT']


# In[134]:

from sklearn.model_selection import train_test_split


# In[135]:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[136]:

from sklearn.linear_model import LinearRegression


# In[137]:

lm = LinearRegression()


# In[138]:

lm.fit(X_train,y_train)


# In[141]:

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[144]:

Datab2.describe()


# In[145]:

y2 = Datab2['SPY']


# In[146]:

X2 = Datab2['XLI']


# In[148]:

import statsmodels.formula.api as smf


# In[150]:

lm2 = smf.ols(formula='y2 ~ X2', data=Datab2).fit()


# In[151]:

lm2.summary()


# In[ ]:



