import pandas as pd
from sklearn.cluster import MiniBatchKMeans
import numpy as np


def cluster_missing(X_train):

	# create cluster and fill in missing values with most_common / median value

	if X_train.shape[1] == 59:
		X = X_train.drop(["id", "target"], axis = 1)

	else:
		X = X_train.drop(["id"], axis = 1)


	# Drop from the data-set the columns with missing values 
	na_count = X.isnull().sum()
	na_columns = list(na_count[na_count>0].index.values)
	X_no_missing = X.drop(na_columns, axis = 1)
 

	cat_columns_no_missing = list(filter(lambda x: x.endswith("cat"),
                                     X_no_missing.columns.values))
	X_no_missing_dummies = pd.get_dummies(X_no_missing, columns = cat_columns_no_missing)   

	#train kmeans
	kmeans = MiniBatchKMeans(n_clusters = 15, random_state = 0, batch_size = 2000)
	kmeans.fit(X_no_missing_dummies)

	X["cluster"] = kmeans.labels_

	# find median or most common value per cluster for missing values
	Values_replace_missing = pd.DataFrame()

	for col in na_columns:
	    clean_df = X[["cluster", col]].dropna()
	    if col.endswith("cat"):
	        Values_replace_missing[col] = clean_df.groupby(["cluster"]).agg(lambda x:x.value_counts().index.values[0])
	    else:
	        Values_replace_missing[col] = clean_df.groupby(["cluster"]).median() 

	#replace missing values with median or most common value in the same cluster
	for cl, cat in ((x, y) for x in range(15) for y in na_columns):
	    X_train.loc[(X["cluster"] == cl) & pd.isnull(X[cat]), cat] = Values_replace_missing.loc[cl, cat]

	return X_train