#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
class KUS:
    ''' 
    cluster function take input as data to be clustered and 
    return the resultant cluster and silhouette score
    '''
    def cluster(self,data):
       
        feature = data[:, :-1]
        target = data[:,-1]
        kmeans = KMeans(n_jobs = -1, n_clusters = 2, init='k-means++')
        kmeans.fit(feature)
        pred = kmeans.predict(feature)
        
        sil_score = silhouette_score(feature, kmeans.labels_, metric='euclidean')
        frame_data = pd.DataFrame(data)
        frame_data['cluster'] = pred
        array_data = frame_data.values
        cluster_0 = array_data[np.nonzero(array_data[:,-1]==0)]
        cluster_0_data = cluster_0[:, :-1]
        cluster_1 = array_data[np.nonzero(array_data[:,-1]==1)]
        cluster_1_data = cluster_1[:, :-1]
        return cluster_0_data, cluster_1_data, sil_score

    '''
    combine function combine all the cluster data with each other,
    it take all the cluster data and return the combined sub-data formed
    '''
    def combine(self, Min_0, Min_1, Maj_0, Maj_1):
        sd_00 = np.concatenate((Min_0, Maj_0))
        sd_01 = np.concatenate((Min_0, Maj_1))
        sd_10 = np.concatenate((Min_1, Maj_0))
        sd_11 = np.concatenate((Min_1, Maj_1))
        return sd_00, sd_01, sd_10, sd_11

    ''' 
    silhouette_largest function take all the subdata formed by combine function as input 
    then apply the cluster function on the sub data, and get the silhouette score for each sub data
    return majority cluster from sub data having the largest silhouette score and the largest silhouette score
     '''
    def silhouette_largest(self, k1, k2, k3, k4):
        X1, Y1, s1 = self.cluster(k1)
        X2, Y2, s2 = self.cluster(k2)
        X3, Y3, s3 = self.cluster(k3)
        X4, Y4, s4 = self.cluster(k4)
        lar_sil = max(s1, s2, s3, s4)
        if(lar_sil == s1):
            majority_cluster_data = k1[np.nonzero(k1[:,-1]==0)]
        elif(lar_sil == s2):
            majority_cluster_data = k2[np.nonzero(k2[:,-1]==0)]
        elif(lar_sil == s3):
            majority_cluster_data = k3[np.nonzero(k3[:,-1]==0)]
        else:
            majority_cluster_data = k4[np.nonzero(k4[:,-1]==0)]
        return  majority_cluster_data, lar_sil

    '''
    KUS function is the main function for performing the k-means undersampling
    it takes training data as input, and sepeare the majority and minority class.
    Then apply cluster function on minority and majority class data. 
    Apply the combine function on the resultant cluster.
    Find the majority cluster with largest silhouette score using silhouette_largest function.
    Return the resultant data formed by joining the selected majority cluster and the minority data.
    '''
    def resample(self,X,Y):
        
        data = np.column_stack([X,Y])
        Maj = data[np.nonzero(data[:,-1]==0)]
        Min = data[np.nonzero(data[:,-1]==1)]
       # print("Min shape ",Min.shape)
       # print("Maj shape ",Maj.shape)
        Min_0, Min_1, sil_min = self.cluster(Min) ## cluster minority
      #  print("Min cluster 0: ",Min_0.shape)
      #  print("Min cluster 1: ",Min_1.shape)
      #  print("sil_min: ",sil_min)
        Maj_0, Maj_1, sil_maj = self.cluster(Maj) ## cluster majority
      #  print("Maj cluster 0: ",Maj_0.shape)
      #  print("Maj cluster 1: ",Maj_1.shape)
      #  print("sil_maj: ",sil_maj)
        sd_00, sd_01, sd_10, sd_11 = self.combine(Min_0,Min_1,Maj_0,Maj_1) ## combine minority and majority clusters with each other
     #   print("sd_00 shape: ",sd_00.shape)
      #  print("sd_01 shape: ",sd_01.shape)
      #  print("sd_10 shape: ",sd_10.shape)
      #  print("sd_11 shape: ",sd_11.shape)
        majority_cluster, lar_sil_score = self.silhouette_largest(sd_00, sd_01, sd_10, sd_11) ##highest silhouette score
      #  print(majority_cluster.shape)
       # print(lar_sil_score)
        training_data = np.concatenate((Min, majority_cluster)) ##combine minority data with majority cluster having highest silhouette score
        X_train_kus, Y_train_kus = training_data[:,:-1], training_data[:,-1]
        return X_train_kus, Y_train_kus


# In[ ]:




