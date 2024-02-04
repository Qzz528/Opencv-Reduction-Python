# -*- coding: utf-8 -*-
"""
OPENCV
threshold

thresh, dst = cv2.threshold(src, thresh, maxval, type)
src:单通道矩阵 
thresh:阈值
maxval:src矩阵数值越过thresh时的设定值
type:处理方式,int 0~4
    0 cv2.THRESH_BINARY     src超过thresh部分值为maxval，否则为0
    1 cv2.THRESH_BINARY_INV src超过thresh部分值为0，否则为maxval
    2 cv2.THRESH_TRUNC      src超过thresh部分值为thresh，否则不变
    3 cv2.THRESH_TOZERO     src超过thresh部分值不变，否则为0
    4 cv2.THRESH_TOZERO_INV src超过thresh部分值为0，否则不变       
RETURN
thresh:输入的阈值
dst:处理后的单通道矩阵
"""

import cv2
import numpy as np

def threshold_numpy(src, thresh, maxval, type):
    dst = np.zeros_like(src) if type <2 else src.copy()
    if type == 0:
        dst[src>thresh] = maxval
    elif type == 1:
        dst[src<=thresh] = maxval
    elif type == 2:
        dst[src>thresh] = thresh
    elif type == 3:
        dst[src<=thresh] = 0
    elif type == 4:
        dst[src>thresh] = 0
    return thresh, dst


def threshold(src, thresh, maxval, type):
    h,w = src.shape
    dst = src.copy()
    for i in range(h):
        for j in range(w):  
            if type == 0:
                dst[i,j] = maxval if dst[i,j]>thresh else 0
            elif type == 1:
                dst[i,j] = 0 if dst[i,j]>thresh else maxval
            elif type == 2:
                dst[i,j] = thresh if dst[i,j]>thresh else dst[i,j]
            elif type == 3:
                dst[i,j] = dst[i,j] if dst[i,j]>thresh else 0
            elif type == 4:
                dst[i,j] = 0 if dst[i,j]>thresh else dst[i,j]
    return thresh, dst


src = np.arange(8).reshape(1,8).astype(float)
print(src)
print('\n')
for i in range(5):
    print(cv2.threshold(src, thresh=3, maxval=10, type=i))
    print(threshold_numpy(src, thresh=3, maxval=10, type=i))   
    print(threshold(src, thresh=3, maxval=10, type=i)) 
    print('\n')


src = cv2.imread("./pics/LenaRGB.bmp")
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)#单通道才能使用
for i in range(5):
    thresh,dst = threshold(gray,thresh=127,maxval=255,type=i)
    cv2.imwrite(f"./pics/threshold/threshold-{i}.jpg",dst)
    thresh,dst = threshold_numpy(gray,thresh=127,maxval=255,type=i)
    cv2.imwrite(f"./pics/threshold/threshold-{i}_np.jpg",dst)
    thresh,dst = cv2.threshold(gray,thresh=127,maxval=255,type=i)
    cv2.imwrite(f"./pics/threshold/threshold-{i}_cv.jpg",dst)
