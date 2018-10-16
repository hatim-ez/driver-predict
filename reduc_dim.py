import numpy as np
import matplotlib.pyplot as plt 
from read_dataset import read_dataset
import math
from imputation import count
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def PCA(data):
	N = len(data)
	scale(data)
	cov = np.dot(np.transpose(data),data) / N
	eig_values, eig_vectors = np.linalg.eig(cov)
	reduced = np.dot(np.transpose(eig_vectors), np.transpose(data))

	eig_values = np.sort(eig_values)
	#eig_values = np.flip(eig_values)
	sum_values = np.sum(eig_values)
	val_max = max(eig_values)
	for i in range(len(eig_values)):
		if eig_values[i] == val_max:
			print ("vector", eig_vectors[i])

	print(eig_values)
	list_for_gap = [e/sum_values for e in eig_values]
	X = range(len(eig_values))
	plt.plot(X, list_for_gap)
	plt.show()

	return reduced




def scale(data):
	print(data.shape)
	for j in range(len(data[:][0])):
		m = np.mean(data[:][j])
		v = np.std(data[:][j])
		data[:][j] = (data[:][j] - m) / math.sqrt(v)
	for j in range(len(date[:][0])):
		m = np.mean(data[:][j])
		v = np.std(data[:][j])
		print(m,v)


def LDA_process(X):
    l,n = X.shape
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X[:,:-1],X[:,-1])
    Y = lda.transform(X[:,:-1])
    return Y

def LDA_sort(Y,y,n0,n1):
    a = np.array([0]*n0)
    b = np.array([0]*n1)
    c=y.shape[0]
    na=0
    nb=0
    for i in range(c):
        if y[i]==0:
            a[na]=Y[i]
            na+=1
        else :
            b[nb]=Y[i]
            nb+=1
    return a,b

def plotting_LDA(a,b):   
    params = dict( alpha=0.3, normed=True)
    plt.hist(a, **params)
    plt.hist(b, **params)
    plt.show()

def LDA(X):
    n0,n1 = count(X)
    Y = LDA_process(X)
    y = X[:,-1]
    a,b = LDA_sort(Y,y,n0,n1)
    plotting_LDA(a,b)
    
