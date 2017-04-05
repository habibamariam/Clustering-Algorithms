# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:58:27 2017

@author: Habiba
"""
%reset -f
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[3,4]].values
              
#using the elbow method t find the no of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('k mean elbow chart')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#applying to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)              

#visulization the clusters
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c='red',label='careful')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c='green',label='standard')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c='blue',label='target')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c='yellow',label='careless')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c='cyan',label='sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,c='magenta',label='centeroids')

plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()
