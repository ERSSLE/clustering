# encoding: utf-8
"""

"""

#author information
__copyright__ = 'Copyright © 2022 ERSSLE'
__license__ = 'MIT License'

import numpy as np
from time import time
from . import tools,kmeans

def center_formula(X,W,p):
    """
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    W: 2D-numpy.array 行代表数据点，列代表类，i行j列代表第i个点隶属于j类的权值
    
    return:
    质心
    """
    return ((X[:,:,np.newaxis] * W[:,np.newaxis,:]**p).sum(0) / (W**p).sum(0,keepdims=True)).T

def weight_update(X,C,p):
    """
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    C: 2D-numpy.array 质心点，每行代表一个质心
    
    return:
    权重
    """
    dst = tools.disfunc(X,C,func='l2')
    dst[dst==0] = 1e-9 # 用一个很小的数据代替0
    p = 1 / max([(p-1),1e-5])
    W = (1/dst)**p
    W = W / W.sum(1,keepdims=True)
    return W,dst

def fcm_alpha(X,n_cluster,p=2,init_method='random',sample=1,seed=None,stop_rate=0.005):
    """
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    n_cluster: 聚类数量
    p: 
    init_method: 初始质心的设定方式，random为随机，scatter根据数据点将初始质心尽可能分散开
    sample: 初始质心确定时，采样X的比例
    stop_rate: 聚类停止条件
    seed: 聚类中心初始化随机种子
    """
    cluster_centers = kmeans.init_centers(X,n_cluster,target_func='l2',method=init_method,sample=sample,seed=seed)
    weights,dst = weight_update(X,cluster_centers,p)
    while True:
        cluster_centers = center_formula(X,weights,p)
        new_weights,dst = weight_update(X,cluster_centers,p)
        max_update = (np.abs(weights - new_weights) / weights).max()
        max_update = np.abs(weights - new_weights).max()
        if max_update <= stop_rate:
            break
        else:
            weights = new_weights
    return weights,cluster_centers,dst

def fcm_func(X,n_cluster,p=2,init_method='random',sample=1,seed=None,stop_rate=0.005,loop=10):
    """
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    n_cluster: 聚类数量
    p:
    init_method: 初始质心的设定方式，random为随机，scatter根据数据点将初始质心尽可能分散开
    sample: 初始质心确定时，采样X的比例
    stop_rate: 聚类停止条件
    seed: 聚类中心初始化随机种子
    loop:循环次数以取最佳结果
    """
    best_score = None
    if seed is None:
        seed = int(time())
    for i in range(loop):
        weights,cluster_centers,dst = fcm_alpha(X,n_cluster,p,init_method,sample,seed+i,stop_rate)
        dst = dst * weights**p
        score = dst.sum()
        if (best_score is None) or (score < best_score):
            best_score = score
            best_weights = weights
            best_cluster_centers = cluster_centers
            best_dst = dst
            best_c = weights.argmax(1)
    return best_score,best_weights,best_cluster_centers,best_dst,best_c

def fcm_predict(X,cluster_centers,p):
    weights,dst = weight_update(X,cluster_centers,p)
    return weights

class FCM():
    def __init__(self,n_cluster,p,init_method='random',sample=1,stop_rate=0.005,seed=None,loop=10):
        """
        n_cluster: 聚类数量
        p:
        init_method: 初始质心的设定方式，random为随机，scatter根据数据点将初始质心尽可能分散开
        sample: 初始质心确定时，采样X的比例
        stop_rate: 聚类停止条件
        seed: 聚类中心初始化随机种子
        loop:循环次数以取最佳结果
        """
        self._n_cluster = n_cluster
        self._p = p
        self._init_method = init_method
        self._sample = sample
        self._stop_rate = stop_rate
        self._seed = seed
        self._loop = loop
    def fit(self,X):
        self.score_,self.weights_,self.cluster_centers_,self.dst_,self.labels_ =\
        fcm_func(X,self._n_cluster,self._p,self._init_method,self._sample,self._seed,
            self._stop_rate,self._loop)
        
    def predict(self,X):
        return fcm_predict(X,self.cluster_centers_,self._p)