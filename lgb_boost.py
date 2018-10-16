# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:30:48 2017

@author: bp
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from gini import *

MAX_ROUNDS = 1200
OPTIMIZE_ROUNDS = True
LEARNING_RATE = 0.024

#prediction with cross validation
def predict_lgboost_k_fold(train_df,test_df,k):
    kf=KFold(n_splits=k)
    results=0
    
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    feature_columns_to_use = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15']
    #pred=[[0,0]]*test_df.shape[0]
    params = {'learning_rate': 0.024, 'max_depth': 5,'lambda_l1': 16.7, 'objective': 'binary', 'metric': 'auc', 'max_bin': 1000, 'feature_fraction': .7, 'is_training_metric': False, 'seed': 99}

    #params = {'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': {'l2', 'auc'},'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,'bagging_fraction': 0.8,'bagging_freq': 5,'verbose': 0}
    y_pred=[0]*test_df.shape[0]
    X_pred=test_df[feature_columns_to_use]
    for train, test in kf.split(train_df):
        train_data = train_df.loc[train]
        test_data = train_df.loc[test]
        X_train = train_data[feature_columns_to_use]
        X_test = test_data[feature_columns_to_use]
        y_train = train_data['target']
        y_test=test_data['target']
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
          
        print('Start training...')
        # train
        model = lgb.train(params,lgb_train,num_boost_round=1200,valid_sets=lgb_eval, verbose_eval=50,early_stopping_rounds=200)
        print('Start predicting...')
        # predict
        y_eval = model.predict(X_test, num_iteration=model.best_iteration)
        y_pred += model.predict(X_pred, num_iteration=model.best_iteration)
        results += gini_lgb(y_eval, test_data)
    return results/k, y_pred/k

#simple lgboost prediction
def predict_lgboost(train_df,test_df):
    feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
    #pred=[[0,0]]*test_df.shape[0]
    params = {'learning_rate': 0.01,'num_iterations':200, 'max_depth': -1, 'boosting': 'dart', 'objective': 'binary', 'metric': 'auc','num_leaves': 50, 'min_data_in_leaf': 200, 'max_bin': 1000,'is_training_metric': False, 'seed': 99}
    #params = {'task': 'train','boosting_type': 'gbdt','objective': 'regression','metric': {'l2', 'auc'},'num_leaves': 31,'learning_rate': 0.05,'feature_fraction': 0.9,'bagging_fraction': 0.8,'bagging_freq': 5,'verbose': 0}
    
    y_pred=[0]*test_df.shape[0]
    X_pred=test_df[feature_columns_to_use]
    X_train = train_df[feature_columns_to_use]
    y_train = train_df['target']
    
    
    lgb_train = lgb.Dataset(X_train, y_train)
    #lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
    gbm = lgb.train(params,lgb_train,num_boost_round=100, verbose_eval=100)
    print('Start predicting...')
    # predict
    y_pred=gbm.predict(X_pred, num_iteration=gbm.best_iteration)
    
    return y_pred
    
