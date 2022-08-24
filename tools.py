# encoding: utf-8
"""

"""

# original author information
__copyright__ = 'Copyright © 2022 ERSSLE'
__license__ = 'MIT License'

import numpy as np

def disfunc_alpha(X,Y,func='l2'):
    if func == 'l2':
        dst = ((X[:,:,np.newaxis] - Y[np.newaxis,:,:].transpose(0,2,1))**2).sum(1) # L2
    elif func == 'l1':
        dst = np.abs(X[:,:,np.newaxis] - Y[np.newaxis,:,:].transpose(0,2,1)).sum(1) # L1
    elif func == 'cosine':
        dst = np.dot(X,Y.T) /\
            (np.sqrt((X**2).sum(1,keepdims=True)) * np.sqrt((Y**2).sum(1,keepdims=True).T)) # cosine
    return dst

def truncate(x,size=1000):
    lenx = len(x)
    for i in range(0,lenx,size):
        if i < lenx:
            yield x[i:i+size]

def disfunc(X,Y,func='l2',size=1000):
    dst = None
    for xi in truncate(X,size):
        y_dst = None
        for yi in truncate(Y,size):
            if y_dst is None:
                y_dst = disfunc_alpha(xi,yi,func)
            else:
                y_dst = np.column_stack((y_dst,disfunc_alpha(xi,yi,func)))
        if dst is None:
            dst = y_dst
        else:
            dst = np.row_stack((dst,y_dst))
    return dst