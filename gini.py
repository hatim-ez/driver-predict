# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 00:01:12 2017

@author: bp
"""
import numpy as np

def gini(actual, pred, cmpcol=0, sortcol=1):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)


def gini_xgb(preds, dtrain):
    labels = dtrain['target']
    gini_score = gini_normalized(labels, preds)
    return gini_score

def gini_lgb(preds, dtrain):
    y = dtrain['target']
    score = gini(y, preds) / gini(y, y)
    return  score