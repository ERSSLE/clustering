# encoding: utf-8
"""
通过测试，本算法实现比sklearn.cluster.KMeans算法要快很多
"""

# original author information
__copyright__ = 'Copyright © 2022 ERSSLE'
__license__ = 'MIT License'

import numpy as np

def k_means(X,n_cluster):
    """k-means算法
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    n_cluster: 聚类数量
    """
    init_cluster_idx = np.random.randint(0,len(X),size=n_cluster)
    cluster_centers = X[init_cluster_idx]
    while True:
        #计算借用numpy的广播机制，这样做会使计算更快速
        dst = np.sqrt(((X[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:].transpose(0,2,1))**2).sum(1))
        c = dst.argmin(1)
        new_cluster_centers = np.empty_like(cluster_centers)
        for i in range(len(cluster_centers)):
            new_cluster_centers[i] = X[c==i].mean(0).tolist()
        if (new_cluster_centers == cluster_centers).sum() == cluster_centers.size:
            break
        else:
            cluster_centers = new_cluster_centers
    return dst,c,cluster_centers

def k_means_predict(X,cluster_centers):
    """根据聚类中心对新数据进行归类"""
    dst = np.sqrt(((X[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:].transpose(0,2,1))**2).sum(1))
    c = dst.argmin(1)
    return c

class KMeans():
    """k-means算法的类创建，基于k_means与k_means_predict两个函数"""
    def __init__(self,n_cluster):
        self._n_cluster = n_cluster
    def fit(self,X):
        _,self.labels_,self.cluster_centers_ = k_means(X,self._n_cluster)
    def predict(self,X):
        return k_means_predict(X,self.cluster_centers_)
