# encoding: utf-8
"""

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

#===============================================================
#下面对kmeans算法的相似性或距离函数进行了扩展，质心也对应进行了扩展，可以依据场景用不同方法进行kmeans聚类
#算法停止条件依据聚类类别在迭代中变化比例是否小于等于阈值
from time import time

def disfunc(X,cluster_centers,target_func='l2'):
    """参数解释见k_means"""
    #这里计算借用numpy的广播机制，这样做会使计算更快速
    if target_func == 'l2':
        dst = ((X[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:].transpose(0,2,1))**2).sum(1) # L2
    elif target_func == 'l1':
        dst = np.abs(X[:,:,np.newaxis] - cluster_centers[np.newaxis,:,:].transpose(0,2,1)).sum(1) # L1
    elif target_func == 'cosine':
        dst = np.dot(X,cluster_centers.T) /\
            (np.sqrt((X**2).sum(1,keepdims=True)) * np.sqrt((cluster_centers**2).sum(1,keepdims=True).T)) # cosine
    return dst

def init_centers(X,n_cluster,target_func='l2',method='random',sample=0.05,seed=None):
    """参数解释见k_means"""
    if seed is None:
        seed = int(time())
    np.random.seed(seed)
    x_samples = np.random.permutation(np.unique(X,axis=0))[:int(len(X)*sample)]
    if method == 'random':
        cluster_centers = np.random.permutation(x_samples)[:n_cluster]
    elif method == 'scatter':
        if target_func in ['l2','cosine']:
            cluster_centers = x_samples.mean(0,keepdims=True)
        elif target_func == 'l1':
            cluster_centers = np.median(x_samples,axis=0,keepdims=True)
        for i in range(n_cluster-1):
            dst = disfunc(x_samples,cluster_centers,target_func)
            if target_func in ['l2','l1']:
                dst_selected = dst[(abs(dst)<1e-5).sum(1)==0]
                x_samples_selected = x_samples[(abs(dst)<1e-5).sum(1)==0]
                next_idx = dst_selected.sum(1).argmax(0) # 选取分散点，注意避免选取相同点
            elif target_func == 'cosine':
                dst_selected = dst[(abs(dst-1.0)<1e-5).sum(1)==0]
                x_samples_selected = x_samples[(abs(dst-1.0)<1e-5).sum(1)==0]
                next_idx = dst_selected.sum(1).argmin(0) # 选取分散点，注意避免选取相同点
            cluster_centers = np.append(cluster_centers,x_samples_selected[[next_idx]],axis=0)
    return cluster_centers

def k_means_alpha(X,n_cluster,target_func='l2',init_method='random',sample=0.05,stop_rate=0.0,seed=None):
    """k-means算法
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    n_cluster: 聚类数量
    target_func:l2,l1,cosine
    init_method: 初始质心的设定方式，random为随机，scatter根据数据点将初始质心尽可能分散开
    sample: 初始质心确定时，采样X的比例
    stop_rate: 聚类停止条件
    seed: 聚类中心初始化随机种子
    """
    cluster_centers = init_centers(X,n_cluster,target_func,init_method,sample,seed)
    c = None
    while True:
        dst = disfunc(X,cluster_centers,target_func)
        if target_func == 'cosine':
            new_c = dst.argmax(1)
        elif target_func in ['l2','l1']:
            new_c = dst.argmin(1)
        rate = (c != new_c).sum() / len(new_c)
        c = new_c
        if rate <= stop_rate:
            break
        else:
            for i in range(len(cluster_centers)):
                if target_func == 'l1':
                    cluster_centers[i] = np.median(X[c==i],axis=0).tolist()
                elif target_func in ['l2','cosine']:
                    cluster_centers[i] = X[c==i].mean(0).tolist() 
    return dst,c,cluster_centers

def k_means2(X,n_cluster,target_func='l2',init_method='random',sample=0.05,stop_rate=0.0,seed=None,loop=10):
    """k-means算法
    X: 2D-numpy.array数据类型,行代表对象，列代表维度
    n_cluster: 聚类数量
    target_func:l2,l1,cosine
    init_method: 初始质心的设定方式，random为随机，scatter根据数据点将初始质心尽可能分散开
    sample: 初始质心确定时，采样X的比例
    stop_rate: 聚类停止条件
    seed: 聚类中心初始化随机种子
    loop: 循环次数以取最佳结果
    """
    best_score = None
    if seed is None:
        seed = int(time())
    for i in range(loop):
        dst,c,cluster_centers = k_means_alpha(X,n_cluster,target_func,init_method,sample,stop_rate,seed+i)
        if target_func == 'cosine':
            score = dst.max(1).mean()
        elif target_func in ['l1','l2']:
            score = dst.min(1).mean()
        if (best_score is None) or (target_func == 'cosine' and score > best_score) or\
                (target_func in ['l2','l1'] and score < best_score):
            best_score = score
            best_dst = dst
            best_c = c
            best_cluster_centers = cluster_centers
    return best_dst,best_c,best_cluster_centers

def k_means_predict2(X,cluster_centers,target_func='l2'):
    """根据聚类中心对新数据进行归类"""
    dst = disfunc(X,cluster_centers,target_func)
    if target_func == 'cosine':
        c = dst.argmax(1)
    elif target_func in ['l2','l1']:
        c = dst.argmin(1)
    return c

class KMeans2():
    """k-means算法的类创建，基于k_means与k_means_predict两个函数"""
    def __init__(self,n_cluster,target_func='l2',init_method='random',sample=0.05,stop_rate=0.0,seed=None,loop=10):
        self._n_cluster = n_cluster
        self._target_func = target_func
        self._init_method = init_method
        self._sample = sample
        self._stop_rate = stop_rate
        self._seed = seed
        self._loop = loop
    def fit(self,X):
        dst,self.labels_,self.cluster_centers_ = k_means2(X,self._n_cluster,self._target_func,
                                self._init_method,self._sample,self._stop_rate,self._seed,self._loop)
        if self._target_func == 'cosine':
            self.avg_score_ = dst.max(1).mean()
        elif self._target_func in ['l2','l1']:
            self.avg_score_ = dst.min(1).mean()
    def predict(self,X):
        return k_means_predict2(X,self.cluster_centers_,self._target_func)
