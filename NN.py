# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:21:51 2017

@author: bp
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from gini import *
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

def log_reg_k_fold(train_df, test_df, k):
    kf=KFold(n_splits=k)
    results=0
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    feature_columns_to_use = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin']


    y_pred=[[0,0]]*test_df.shape[0]
    X_pred=test_df[feature_columns_to_use]
    
    for train, test in kf.split(train_df):
        train_data = train_df.loc[train]
        test_data = train_df.loc[test]
        X_train = train_data[feature_columns_to_use]
        X_test = test_data[feature_columns_to_use]
        y_train = train_data['target']
        y_test=test_data['target']
        
        # train
        model = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30, 10), random_state=1, n_iter=200)
        print('Start training...')
        # train
        model.fit(X_train, y_train) 
        print('Start predicting...')
        # predict
        y_eval = model.predict_proba(X_test)
        y_pred += model.predict_proba(X_pred)

        results += gini_lgb(y_eval, test_data)
    return results/k, y_pred/k