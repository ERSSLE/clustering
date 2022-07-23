# encoding: utf-8
"""
层次聚类在scipy与sklearn当中都有实现，这里不再进行具体实现。
层次聚类根据邻近度度量方式的不同，有single（单链）,complete（全链）,average（均值）,ward等方法
sklearn当中没有对层次聚类可视化的实现，scipy当中有dendrogram，用以下to_linkage_matrix可以将sklearn拟合结果转化为
scipy可以可视化的linkage matrix.
"""

# original author information
__copyright__ = 'Copyright © 2022 ERSSLE'
__license__ = 'MIT License'

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

def to_linkage_matrix(agg):
    if isinstance(agg,AgglomerativeClustering) and 'distances_' in dir(agg):
        children = agg.children_
        distances = agg.distances_[:,np.newaxis]
        length = len(agg.labels_)
        def count_node(child,num=0):
            for node in child:
                if node < length:
                    num += 1
                else:
                    num = count_node(children[node-length],num)
            return num
        nums = []
        for child in children:
            nums.append(count_node(child))
        nums = np.array(nums)[:,np.newaxis]
        return np.column_stack((children,distances,nums))
    else:
        raise ValueError('agg 必须为 sklearn.cluster.AgglomerativeClustering 且其distances_有值')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    points = np.array([
    [0.4005,0.5306],
    [0.2148,0.3854],
    [0.3457,0.3156],
    [0.2652,0.1875],
    [0.0789,0.4139],
    [0.4548,0.3022]
    ])
    agg = AgglomerativeClustering(compute_distances=True)
    agg.fit(points)
    z = to_linkage_matrix(agg)
    _ = dendrogram(z)
    plt.show()