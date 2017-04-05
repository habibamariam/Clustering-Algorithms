# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:16:15 2017

@author: Habiba
"""

#%reset -f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv('Mall_Customers.csv')

X=dataset.iloc[:,[3,4]].values
#using dendogram for right no of clusters
import scipy.cluster.hierarchy as sch

dendrogram=sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('dendrogram')
plt.xlabel('customers')
plt.ylabel('euclidian distances')

#fiting the HC to the dataset
from  sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(X)

#plotting the clusters
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='yellow',label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='cyan',label='sensible')

plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()