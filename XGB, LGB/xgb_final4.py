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
import xgboost as xgb


# In[2]:


train=pd.read_csv('train_final4.csv')
test=pd.read_csv('test_final4.csv')


# In[3]:


train.dtypes


# In[5]:


obj_cols =train.dtypes
obj_cols[obj_cols=='object']


# In[6]:


train['card2_and_count']=pd.to_numeric(train['card2_and_count'],errors='coerce')
train['addr1_card1']=pd.to_numeric(train['addr1_card1'],errors='coerce')


# In[7]:


test['card2_and_count']=pd.to_numeric(test['card2_and_count'],errors='coerce')
test['addr1_card1']=pd.to_numeric(test['addr1_card1'],errors='coerce')


# In[8]:


split_groups = train['DT_M']


# In[9]:


X= train.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
               'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)
y = train['isFraud']


# In[10]:


X_test = test.drop(['TransactionDT','TransactionID','uid','uid2','bank_type',
                    'isFraud','DT','DT_M','DT_W','DT_D','DT_hour','DT_day_week','DT_day_month'], axis=1)


# In[11]:


X_test.shape


# In[13]:


X.shape


# In[14]:


del train, test
gc.collect()


# In[16]:


print(X.shape, X_test.shape)


# In[10]:


#fill in -999 for alll missing
#X = X.fillna(-999)
#X_test = X_test.fillna(-999)


# In[51]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[53]:


columns = X.columns
splits = folds.split(X, y, groups=split_groups)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importance_df = pd.DataFrame()
dtest = xgb.DMatrix(X_test)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    dtrain = xgb.DMatrix(X[columns].iloc[train_index], y.iloc[train_index])
    dvalid = xgb.DMatrix(X[columns].iloc[valid_index], y.iloc[valid_index])
    y_valid= y.iloc[valid_index]
    params = {'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'booster':'gbtree',
              'nthread' : 4,
              'eta' : 0.048,
              'max_depth' : 9,
              'missing': -999,
              #'tree_method':'gpu_hist',
              'subsample' : 0.85,
              'colsample_bytree' : 0.85,
              'alpha' : 0.15,
              'lambda' : 0.85,
              'random_state': 99, 
              'silent': True}
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
    model=xgb.train(params, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=200)
    
    y_pred_valid = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += model.predict(dtest,ntree_limit=model.best_ntree_limit) / NFOLDS
    fold_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame(list(model.get_fscore().keys()), columns=['feature'])
    fold_importance_df['importance']=list(model.get_fscore().values())
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del model, dtrain, dvalid, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[57]:


feature_importances = feature_importance_df.groupby('feature').agg({'importance':np.mean}).reset_index()


# In[62]:


feature_importances.head()


# In[60]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='importance', ascending=False).head(50), x='importance', y='feature');
plt.title('50 TOP feature importance over 5 folds average');


# In[61]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='importance', ascending=False).tail(50), x='importance', y='feature');
plt.title('50 tail feature importance');


# In[54]:


sub = pd.read_csv('sample_submission.csv')


# In[55]:


sub['isFraud'] = y_preds
sub.to_csv("xgb_final4.csv", index=False) #0.9460


# In[64]:


feature_importances['perc']=feature_importances['importance']/np.sum(feature_importances['importance'])


# In[65]:


feature_importances['cum_sum'] = feature_importances['importance'].cumsum()
feature_importances['cum_perc'] = 100*feature_importances['cum_sum']/feature_importances['importance'].sum()


# In[66]:


feature_importances['cum_perc']


# In[78]:


selected_feature=feature_importances[feature_importances['cum_perc']<=95]['feature']


# In[79]:


len(selected_feature)


# In[80]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[81]:


X2=X[selected_feature]


# In[82]:


X2.shape


# In[83]:


X_test2=X_test[selected_feature]


# In[84]:


X_test2.shape


# In[85]:


columns = X2.columns
splits = folds.split(X2, y, groups=split_groups)
y_preds = np.zeros(X_test2.shape[0])
y_oof = np.zeros(X2.shape[0])
score = 0

feature_importance_df = pd.DataFrame()
dtest = xgb.DMatrix(X_test2)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    dtrain = xgb.DMatrix(X2[columns].iloc[train_index], y.iloc[train_index])
    dvalid = xgb.DMatrix(X2[columns].iloc[valid_index], y.iloc[valid_index])
    y_valid= y.iloc[valid_index]
    params = {'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'booster':'gbtree',
              'nthread' : 4,
              'eta' : 0.048,
              'max_depth' : 9,
              'missing': -999,
              #'tree_method':'gpu_hist',
              'subsample' : 0.85,
              'colsample_bytree' : 0.85,
              'alpha' : 0.15,
              'lambda' : 0.85,
              'random_state': 99, 
              'silent': True}
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
    model=xgb.train(params, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=200)
    
    y_pred_valid = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += model.predict(dtest,ntree_limit=model.best_ntree_limit) / NFOLDS
    fold_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame(list(model.get_fscore().keys()), columns=['feature'])
    fold_importance_df['importance']=list(model.get_fscore().values())
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del model, dtrain, dvalid, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[86]:


sub['isFraud'] = y_preds
sub.to_csv("xgb_final4_re.csv", index=False) #0.9456, no need to remove


# In[87]:


del X2, X_test2
gc.collect()


# In[ ]:


# bayes


# In[113]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)
splits = folds.split(X, y, groups=split_groups)


# In[114]:


for fold_n, (train_index, valid_index) in enumerate(splits):
    train_index_df=pd.DataFrame(train_index,columns=['train_index'])
    valid_index_df=pd.DataFrame(valid_index,columns=['valid_index'])
    del train_index, valid_index
    gc.collect()


# In[115]:


train_index_df.shape


# In[116]:


valid_index_df.shape


# In[117]:


from bayes_opt import BayesianOptimization


# In[118]:


train_index=train_index_df['train_index']
valid_index=valid_index_df['valid_index']


# In[119]:


X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]


# In[120]:


X_train.shape


# In[121]:


X_valid.shape


# In[122]:


y_train.shape


# In[123]:


y_valid.shape


# In[124]:


def XGB_bayesian(
    gamma,
    max_leaves, 
    subsample,
    colsample_bytree,
    min_child_weight,
    max_depth,
    alpha,
    reg_lambda
     ):
    
    # LightGBM expects next three parameters need to be integer. 
    max_leaves = int(max_leaves)
    max_depth = int(max_depth)

    assert type(max_leaves) == int
    assert type(max_depth) == int
    y_oof = np.zeros(X_valid.shape[0])

    param = { 'gamma':gamma,
              'max_leaves': max_leaves, 
              'min_child_weight': min_child_weight,
              'subsample' : subsample,
              'colsample_bytree' : colsample_bytree,
              'max_depth': max_depth,
              'alpha': alpha,
              'reg_lambda': reg_lambda,
              'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'booster':'gbtree',
              'missing': -999,
              'nthread' : 4,
              'eta' : 0.048,
              'random_state': 99, 
              'silent': True}    
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
    model=xgb.train(param, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=200)
    
    y_oof = model.predict(dvalid, ntree_limit=model.best_ntree_limit)

    score = roc_auc_score(y_valid, y_oof)

    return score


# In[125]:


bounds_XGB = {
    'gamma': (0,1),
    'max_leaves': (100, 600), 
    'subsample' : (0.2,0.9),
    'colsample_bytree' : (0.2,0.9),
    'min_child_weight': (0.01, 0.1),   
    'alpha': (0.3, 1), 
    'reg_lambda': (0.3, 1),
    'max_depth':(0,15),
}


# In[126]:


XGB_BO = BayesianOptimization(XGB_bayesian, bounds_XGB)


# In[127]:


print(XGB_BO.space.keys)


# In[128]:


init_points = 3
n_iter = 7


# In[129]:


print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    XGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[130]:


params = {'max_leaves': int(XGB_BO.max['params']['max_leaves']),
          'min_child_weight': XGB_BO.max['params']['min_child_weight'],
          'subsample': XGB_BO.max['params']['subsample'],
          'colsample_bytree': XGB_BO.max['params']['colsample_bytree'],
          'max_depth': int(XGB_BO.max['params']['max_depth']),
          'alpha': XGB_BO.max['params']['alpha'],
          'reg_lambda':XGB_BO.max['params']['reg_lambda'],
          'eval_metric': 'auc',
          'objective': 'binary:logistic',
          'booster':'gbtree',
          'missing': -999,
          'nthread' : 4,
          'eta' : 0.048,
          'random_state': 99, 
          'silent': True
         }


# In[131]:


params


# In[132]:


columns = X.columns
splits = folds.split(X, y, groups=split_groups)
y_preds = np.zeros(X_test.shape[0])
y_oof = np.zeros(X.shape[0])
score = 0

feature_importance_df = pd.DataFrame()
dtest = xgb.DMatrix(X_test)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    dtrain = xgb.DMatrix(X[columns].iloc[train_index], y.iloc[train_index])
    dvalid = xgb.DMatrix(X[columns].iloc[valid_index], y.iloc[valid_index])
    y_valid= y.iloc[valid_index]
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
    model=xgb.train(params, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=200)
    
    y_pred_valid = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += model.predict(dtest,ntree_limit=model.best_ntree_limit) / NFOLDS
    fold_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame(list(model.get_fscore().keys()), columns=['feature'])
    fold_importance_df['importance']=list(model.get_fscore().values())
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del model, dtrain, dvalid, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[134]:


sub['isFraud'] = y_preds
sub.to_csv("xgb_final4_baye.csv", index=False) #0.9471


# In[137]:


feature_importances = feature_importance_df.groupby('feature').agg({'importance':np.mean}).reset_index()


# In[138]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='importance', ascending=False).head(50), x='importance', y='feature');
plt.title('50 TOP feature importance over 5 folds average');


# In[139]:


plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='importance', ascending=False).tail(50), x='importance', y='feature');
plt.title('50 Tail feature importance');


# In[140]:


feature_importances['cum_sum'] = feature_importances['importance'].cumsum()
feature_importances['cum_perc'] = 100*feature_importances['cum_sum']/feature_importances['importance'].sum()


# In[141]:


selected_feature=feature_importances[feature_importances['cum_perc']<=95]['feature']


# In[142]:


len(selected_feature)


# In[143]:


from sklearn.model_selection import GroupKFold
NFOLDS = 5
folds = GroupKFold(n_splits=NFOLDS)


# In[144]:


X2=X[selected_feature]


# In[145]:


X2.shape


# In[146]:


X_test2=X_test[selected_feature]


# In[147]:


X_test2.shape


# In[148]:


params = {'max_leaves': int(XGB_BO.max['params']['max_leaves']),
          'min_child_weight': XGB_BO.max['params']['min_child_weight'],
          'subsample': XGB_BO.max['params']['subsample'],
          'colsample_bytree': XGB_BO.max['params']['colsample_bytree'],
          'max_depth': int(XGB_BO.max['params']['max_depth']),
          'alpha': XGB_BO.max['params']['alpha'],
          'reg_lambda':XGB_BO.max['params']['reg_lambda'],
          'eval_metric': 'auc',
          'objective': 'binary:logistic',
          'booster':'gbtree',
          'missing': -999,
          'nthread' : 4,
          'eta' : 0.048,
          'random_state': 99, 
          'silent': True
         }


# In[149]:


columns = X2.columns
splits = folds.split(X2, y, groups=split_groups)
y_preds = np.zeros(X_test2.shape[0])
y_oof = np.zeros(X2.shape[0])
score = 0

feature_importance_df = pd.DataFrame()
dtest = xgb.DMatrix(X_test2)
  
for fold_n, (train_index, valid_index) in enumerate(splits):
    dtrain = xgb.DMatrix(X2[columns].iloc[train_index], y.iloc[train_index])
    dvalid = xgb.DMatrix(X2[columns].iloc[valid_index], y.iloc[valid_index])
    y_valid= y.iloc[valid_index]
    params = {'eval_metric': 'auc',
              'objective': 'binary:logistic',
              'booster':'gbtree',
              'nthread' : 4,
              'eta' : 0.048,
              'max_depth' : 9,
              'missing': -999,
              #'tree_method':'gpu_hist',
              'subsample' : 0.85,
              'colsample_bytree' : 0.85,
              'alpha' : 0.15,
              'lambda' : 0.85,
              'random_state': 99, 
              'silent': True}
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
        
    model=xgb.train(params, dtrain, 5000, watchlist, maximize=True, early_stopping_rounds = 200, verbose_eval=200)
    
    y_pred_valid = model.predict(dvalid, ntree_limit=model.best_ntree_limit)
    y_oof[valid_index] = y_pred_valid
    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")
    
    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS
    y_preds += model.predict(dtest,ntree_limit=model.best_ntree_limit) / NFOLDS
    fold_importance_df = pd.DataFrame()
    fold_importance_df = pd.DataFrame(list(model.get_fscore().keys()), columns=['feature'])
    fold_importance_df['importance']=list(model.get_fscore().values())
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    del model, dtrain, dvalid, y_valid
    gc.collect()
    
print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[150]:


sub['isFraud'] = y_preds
sub.to_csv("xgb_final4_baye_re.csv", index=False) #0.9461 no need to remove


# In[ ]:




