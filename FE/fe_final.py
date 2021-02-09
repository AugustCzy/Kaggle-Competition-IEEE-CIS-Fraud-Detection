#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import datetime 
import lightgbm as lgb

import os
import gc


# In[2]:


train=pd.read_csv('train_total.csv')
test=pd.read_csv('test_total.csv')


# In[3]:


train.shape


# In[4]:


X_train=pd.read_csv('train_fe4.csv')
X_test=pd.read_csv('test_fe4.csv')


# In[5]:


X_train.shape


# In[6]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[7]:


new_col=list(X_train_not_train)


# In[8]:


len(new_col)


# In[9]:


train[new_col]=X_train[new_col]


# In[10]:


train.shape


# In[11]:


test[new_col]=X_test[new_col]


# In[12]:


test.shape


# In[14]:


X_train=pd.read_csv('trian_xgb.csv')
X_test=pd.read_csv('test_xgb.csv')


# In[15]:


X_train.shape


# In[16]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[17]:


new_col=list(X_train_not_train)


# In[18]:


len(new_col)


# In[19]:


train[new_col]=X_train[new_col]


# In[20]:


train.shape


# In[21]:


test[new_col]=X_test[new_col]


# In[22]:


test.shape


# In[24]:


X_train=pd.read_csv('train_more.csv')
X_test=pd.read_csv('test_more.csv')


# In[25]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[26]:


new_col=list(X_train_not_train)


# In[27]:


len(new_col)


# In[28]:


train[new_col]=X_train[new_col]


# In[29]:


train.shape


# In[30]:


test[new_col]=X_test[new_col]


# In[31]:


test.shape


# In[33]:


X_train=pd.read_csv('train_exp.csv')
X_test=pd.read_csv('test_exp.csv')


# In[34]:


X_train.shape


# In[35]:


train_cols = train.columns
X_train_cols = X_train.columns

X_train_not_train = X_train_cols.difference(train_cols)


# In[36]:


new_col=list(X_train_not_train)


# In[37]:


len(new_col)


# In[38]:


train[new_col]=X_train[new_col]


# In[39]:


train.shape


# In[40]:


test[new_col]=X_test[new_col]


# In[41]:


test.shape


# In[43]:


train.to_csv('train_final4.csv',index=False)
test.to_csv('test_final4.csv',index=False)


# In[ ]:





# In[ ]:




