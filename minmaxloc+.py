# -*- coding: utf-8 -*-
"""
OPENCV
minMaxLoc

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(src)
src:单通道矩阵
Return
min_val:矩阵最小值
max_val:矩阵最大值
min_loc:矩阵中最小值的位置
max_loc:矩阵中最大值的位置
"""

import cv2
import numpy as np

#借助numpy复现minMaxLoc功能
def minmaxloc_numpy(src):
    min_val = np.min(src)
    max_val = np.max(src)
    min_loc = np.argwhere(src==min_val)[0] #只需要返回一个位置
    max_loc = np.argwhere(src==max_val)[0]
    return min_val, max_val, tuple(min_loc), tuple(max_loc)

#不依赖其他库，复现minMaxLoc功能
def minmaxloc(src):
    assert len(src.shape)==2
    h,w = src.shape
    
    #初始化最大值最小值，下列值必定可以被更新
    min_val = float('inf')
    max_val = -float('inf')
    
    for i in range(h):
        for j in range(w):
            if src[i,j] > max_val:
                max_val = src[i,j]
                max_loc = (i,j)
            if src[i,j] < min_val:
                min_val = src[i,j]
                min_loc = (i,j)
                
    return min_val, max_val, min_loc, max_loc



if __name__ == '__main__':   
    #测试opencv和自写方法的一致性
    
    src = np.arange(16).reshape(4,4).astype(float)
    print(src)
    print(cv2.minMaxLoc(src))
    print(minmaxloc_numpy(src)) 
    print(minmaxloc(src))
    
    print('\n')

    src = -256.7*np.ones(16).reshape(4,4)
    print(src)
    print(cv2.minMaxLoc(src))
    print(minmaxloc_numpy(src))  
    print(minmaxloc(src))
  