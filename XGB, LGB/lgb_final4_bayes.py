#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys, gc, warnings, random, datetime


# In[2]:


train=pd.read_csv('train_final4.csv')
test=pd.read_csv('test_final4.csv')


# In[3]:


obj_cols = train.dtypes
obj_cols[obj_cols=='object']


# In[4]:


train['card2_and_count']=pd.to_numeric(train['card2_and_count'],errors='coerce')
train['addr1_card1']=pd.to_numeric(train['addr1_card1'],errors='coerce')


# In[5]:


test['card2_and_count']=pd.to_numeric(test['card2_and_count'],errors='coerce')
test['addr1_card1']=pd.to_numeric(test['addr1_card1'],errors='coerce')


# In[6]:


split_groups = train['DT_M']


# In[7]:


X= train.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
               'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)
y = train['isFraud']


# In[8]:


X.shape


# In[9]:


X_test = test.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
                    'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)


# In[10]:


X_test.shape


# In[11]:


del train, test
gc.collect()


# In[12]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)
splits = folds.split(X, y, groups=split_groups)


# In[13]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    train_index_df=pd.DataFrame(train_index,columns=['train_index'])
    valid_index_df=pd.DataFrame(valid_index,columns=['valid_index'])
    del train_index, valid_index
    gc.collect()


# In[14]:


train_index_df.shape


# In[15]:


valid_index_df.shape


# In[16]:


#no_contribution_feature:根据上次结果而来，不一定是这些


# In[17]:


no_contribution_feature=['V117', 'R_isproton', 'V119', 'V68', 'V305', 'NA_V12_V34','addr1_card1','V241','id_27',
                         'V240','V120','NA_V138_V166','V89','NA_V75_V94','V107','P_isproton','V27','V122',
                         'id_35_count_dist','NA_V95_V137','NA_V53_V74','id_22__count_encoding',
                         'NA_V35_V54','NA_V322_V339','NA_V279_V321','id_27__count_encoding','NA_V1_V11',
                         'NA_V167_V216','V28']


# In[18]:


X2=X.drop(no_contribution_feature, axis=1)


# In[19]:


X2.shape


# In[20]:


X_test2=X_test.drop(no_contribution_feature, axis=1)


# In[21]:


X_test2.shape


# In[22]:


del X, X_test
gc.collect()


# In[23]:


from bayes_opt import BayesianOptimization


# In[36]:


train_index=train_index_df['train_index']
valid_index=valid_index_df['valid_index']


# In[37]:


X_train, X_valid = X2.iloc[train_index], X2.iloc[valid_index]
y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]


# In[38]:


X_train.shape


# In[39]:


X_valid.shape


# In[40]:


y_train.shape


# In[41]:


y_valid.shape


# In[42]:


def LGB_bayesian(
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    min_child_weight,
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    num_leaves = int(num_leaves)
    min_data_in_leaf = int(min_data_in_leaf)
    max_depth = int(max_depth)

    assert type(num_leaves) == int
    assert type(min_data_in_leaf) == int
    assert type(max_depth) == int
    y_oof = np.zeros(X_valid.shape[0])

    param = {
              'num_leaves': num_leaves, 
              'min_data_in_leaf': min_data_in_leaf,
              'min_child_weight': min_child_weight,
              'bagging_fraction' : bagging_fraction,
              'feature_fraction' : feature_fraction,
              'max_depth': max_depth,
              'reg_alpha': reg_alpha,
              'reg_lambda': reg_lambda,
              'objective': 'binary',
              'max_bin':255,
              'random_state': 47,
              'learning_rate': 0.005,
              'boosting_type': 'gbdt',
              'bagging_seed': 11,
              'tree_learner':'serial',
              'verbosity': -1,
              'metric':'auc'}    
    
    trn_data= lgb.Dataset(X_train, label=y_train)
    val_data= lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(param, trn_data,  num_boost_round=10000, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 500)
    
    y_oof = clf.predict(X_valid, num_iteration=clf.best_iteration)  
    
    score = roc_auc_score(y_valid, y_oof)

    return score


# In[43]:


bounds_LGB = {
    'num_leaves': (400, 600), 
    'min_data_in_leaf': (50,150),
    'bagging_fraction' : (0.2,0.9),
    'feature_fraction' : (0.2,0.9),
    'min_child_weight': (0.01, 0.1),   
    'reg_alpha': (0.3, 1), 
    'reg_lambda': (0.3, 1),
    'max_depth':(-1,15),
}


# In[44]:


LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB)


# In[45]:


print(LGB_BO.space.keys)


# In[46]:


init_points = 3 #5
n_iter = 7 # 15


# In[47]:


print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[48]:


params = {'num_leaves': int(LGB_BO.max['params']['num_leaves']),
          'min_child_weight': LGB_BO.max['params']['min_child_weight'],
          'feature_fraction': LGB_BO.max['params']['feature_fraction'],
          'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],
          'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
          'objective': 'binary',
          'tree_learner':'serial',
          'max_depth': int(LGB_BO.max['params']['max_depth']),
          'max_bin':255,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': LGB_BO.max['params']['reg_alpha'],
          'reg_lambda':LGB_BO.max['params']['reg_lambda'],
          'random_state': 47,
         }


# In[49]:


params


# In[50]:


NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[51]:


columns = X2.columns
splits = folds.split(X2, y, groups=split_groups)
y_preds = np.zeros(X_test2.shape[0])
y_oof = np.zeros(X2.shape[0])
score = 0

feature_importances = pd.DataFrame()
feature_importances['feature'] = columns
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    X_train, X_valid = X2[columns].iloc[train_index], X2[columns].iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(params, dtrain, 10000, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)
    
    feature_importances[f'fold_{fold_n + 1}'] = clf.feature_importance()
    
    y_pred_valid = clf.predict(X_valid)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += clf.predict(X_test2) / NFOLDS
    
    del X_train, X_valid, y_train, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[52]:


sub = pd.read_csv('sample_submission.csv')
sub['isFraud'] = y_preds
sub.to_csv("lgb_final4_bayes.csv", index=False) #0.9496


# In[ ]:





# In[ ]:




