# This script shows you how to make a submission using a few
# useful Python libraries.
# It gets a public leaderboard score of 0.76077.
# Maybe you can tweak it and do better...?

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import KFold
from gini import *
import lightgbm as lgb
    
#simple prediction with default parameters for the xgb model
def predict_xgboost(train_df, test_df):
    # Prepare the inputs for the model
    feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']

    train_X = train_df[feature_columns_to_use]
    test_X = test_df[feature_columns_to_use]
    train_y = train_df['target']
        
   
        
    gbm = xgb.XGBClassifier().fit(train_X, train_y)
        
    predictions = gbm.predict_proba(test_X)
    
    return predictions

#xgb prediction with k-fold cross validation and customs parameters  
def predict_xgboost_k_fold(data,k,test_v):       
    
    kf=KFold(n_splits=k)
    results=[]
	
    feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15']
    pred=[[0,0]]*test_v.shape[0]
    
    # Prepare the inputs for the model, with the different corss validation folds
    test_v=test_v[feature_columns_to_use]
    for train, test in kf.split(data):
        train_data = data.loc[train]
        test_data = data.loc[test]
        train_X = train_data[feature_columns_to_use]
        test_X = test_data[feature_columns_to_use]
        train_y = train_data['target']
        #train the xgboost
        gbm = xgb.XGBClassifier(learning_rate= 0.02, max_depth= 4, subsample= 0.9, colsample_bytree= 0.9, objective= 'binary:logistic', seed= 99, silent= True,  n_estimators=300)
        gbm=gbm.fit(train_X, train_y)
        #predict
        predictions = gbm.predict_proba(test_X)
        #compute the gini score
        results.append(gini_xgb(predictions, test_data)) 
        pred+=gbm.predict_proba(test_v)
    
    
    return np.mean(results),pred/k

def Xgb_and_Lgb(train_df, test_df) :
    k=5
    kf=KFold(n_splits=k)
    results=0
    #feature_columns_to_use = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
    feature_columns_to_use = ['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03', 'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin', 'ps_ind_08_bin', 'ps_ind_09_bin', 'ps_ind_14', 'ps_ind_15', 'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01', 'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat', 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat', 'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14', 'ps_car_15', 'ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin']

    #params = {'learning_rate': 0.01,'num_iterations':5000, 'max_depth': -1, 'objective': 'binary', 'metric': 'auc', 'max_bin': 1000,'is_training_metric': False}
    params = {
        'objective':'binary:logistic',        
        'max_depth':5,
        'learning_rate':0.07,
        'eval_metric':'auc',
        'min_child_weight':6,
        'subsample':0.8,
        'colsample_bytree':0.8,
        'reg_lambda':1.3,
        'reg_alpha':8,
        'gamma':10,
        'scale_pos_weight':1.6
        #'n_thread':-1
    }
    
    params2 = {'learning_rate': 0.024, 'max_depth': 5,'lambda_l1': 16.7, 'objective': 'binary', 'metric': 'auc', 'max_bin': 1000, 'feature_fraction': .7, 'is_training_metric': False, 'seed': 99}
    y_pred=[0]*test_df.shape[0]
    X_pred=test_df[feature_columns_to_use]
    #prepare the inputs
    for train, test in kf.split(train_df):
        train_data = train_df.loc[train]
        test_data = train_df.loc[test]
        X_train = train_data[feature_columns_to_use]
        X_test = test_data[feature_columns_to_use]
        y_train = train_data['target']
        y_test=test_data['target']
        #input with xgb library
        xgb_train = xgb.DMatrix(X_train, y_train)
        xgb_eval = xgb.DMatrix(X_test, y_test)
        watchlist = [(xgb_train,'train'),(xgb_eval,'test')]  
        
        #input with lgb library
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
          
        print('Start training lgb...')
        # train
        model = lgb.train(params2,lgb_train,num_boost_round=1200,valid_sets=lgb_eval, verbose_eval=100,early_stopping_rounds=50)
        print('Start training xgb...')
        # train
        gbm = xgb.train(params,xgb_train,1200,watchlist,early_stopping_rounds=50,verbose_eval=50, maximize=True)
        print('Start predicting...')
        # predict with xgb
        y_eval = gbm.predict(xgb.DMatrix(X_test),ntree_limit=gbm.best_ntree_limit+50)
        y_pred += gbm.predict(xgb.DMatrix(X_pred),ntree_limit=gbm.best_ntree_limit+50)
        
        #predict with lgb
        y_eval += model.predict(X_test, num_iteration=model.best_iteration)
        y_pred += model.predict(X_pred, num_iteration=model.best_iteration)
        
        results += gini_lgb(y_eval/2, test_data)
        
    return results/k, y_pred/(2*k)

def submit(predictions, test_df):
    submission = pd.DataFrame({ 'id': [int(a) for a in test_df['id']],'target': predictions[:] })
    print(submission)
    submission.to_csv("submission.csv", index=False) 
    

