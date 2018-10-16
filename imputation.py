# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 19:33:53 2017

@author: bp
"""
import numpy as np
import pandas as pd

def count(data):
    #compte la répartition des données en 0,1
    temp0 = 0
    temp1 = 0
    l,n=data.shape
    for i in range (l):
        if (data[i][n-1]==0):
            temp0 += 1
        else:
            temp1 += 1
    return (temp0,temp1)


def mean_computation(a):
    #prepare the training data (target column")
    #needs 
    data = a.drop(['id'], axis = 1)
    data.replace(-1, np.nan)
    if (data.shape[1]==58):
        means=data.groupby('target').mean()
    else :
        means=data.mean()
    data.replace(np.nan,-1)
    return means



def mean_imputation(a):
    data=a.values
    means = mean_computation(a).values

    l,n=data.shape
    if (n==59): 
       for i in range (l):
           for j in range(2,n) :
                if data[i,j] == -1:
                    cat=int(data[i,1])
                    data[i,j] = means[cat,j-2]
            
    else :
        for i in range (l):
            for j in range(1,n):
                if data[i,j] == -1:
                    data[i,j] = means[j-1]
    test=pd.DataFrame(data)
    test.columns=a.columns
    return test