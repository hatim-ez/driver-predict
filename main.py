from read_dataset import read_dataset
from imputation import *
from reduc_dim import *
from predict_xgboost import *
from clustering_missing import *


print("preprocessing...")
data_train = read_dataset('train.csv')
data_test_ini = read_dataset('test.csv')
#data2_train_modified = mean_imputation(data_train)
#data_test_modified = mean_imputation(data_test_ini)
data_train_modified = cluster_missing(data_train)
data_test_modified = cluster_missing(data_test_ini)

print("processing...")
s1,y1=Xgb_and_Lgb(data_train,data_test_ini)
#s2,y2=Xgb_and_Lgb(data_modified,data_modified)
#predict_xgboost_k_fold(data_train,data_test_ini,5)
#predict_lgboost_k_fold(data_train,data_test_ini,5)
#predict_xgboost_k_fold(data_modified,data_modified,5)
#predict_lgboost_k_fold(data_modified,data_modified,5)