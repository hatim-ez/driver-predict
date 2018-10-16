# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:54:00 2017

@author: bp
"""
import seaborn as sns
import matplotlib.pyplot as plt


def corr_heatmap(data):
    correlations = data.corr()

    # Create color map ranging between two colors
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, cmap=cmap, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, cbar_kws={"shrink": .75})
    plt.show();