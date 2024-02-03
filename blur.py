# -*- coding: utf-8 -*-
"""
blur
对图像进行均值滤波
dst = cv2.blur(src,ksize)
src:图像矩阵（支持多通道）
ksize:双元素tuple，(kh,kw)，分别为滤波器核的高和宽
RETURN
dst:填充边缘后的图像矩阵
"""

import numpy as np
import cv2
from copyMakeBorder import copyMakeBorder_numpy, copyMakeBorder



def blur(src,ksize,borderType=4,value=0,srd = True): #**kwargs
    kh,kw = ksize
    h,w = src.shape[:2]
    # ph = kh-1
    # pw = kw-1
    c = src.shape[2] if len(src.shape)==3 else 0
    # pad_src = np.zeros((h+kh-1,w+kw-1))

    if srd:#四周填充
        top = (kh-1)//2
        bottom = kh-1-top
        left = (kw-1)//2
        right = kw-1-left
    else: #左上方填充
        top = kh-1
        left = kw-1
        bottom = 0
        right = 0
    
    #先扩充src确保运算后dst大小与原src一致
    pad_src = copyMakeBorder(src,top,bottom,left,right,borderType,value)
    
    
    kernel = 1/(kh*kw)*np.ones((kh,kw))
    dst = np.zeros_like(src)
    
    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    dst[i,j,k] = round(np.sum(pad_src[i:i+kh,j:j+kw,k]*kernel))
            else:
                dst[i,j] = round(np.sum(pad_src[i:i+kh,j:j+kw]*kernel))
    return dst

if __name__ == '__main__':
    
    x = np.arange(1,10).reshape(3,3)
    #y = np.dstack((x,x,x))
    

    
    print(cv2.blur(x,(2,2)))
    print(blur(x,(2,2),srd=False))
    print('\n')

    for i in range(5):
        print(cv2.blur(x,(2,2),borderType=1))
        print(blur(x,(2,2),borderType=1,srd=False))
        print('\n')

    
    #print(cv2.blur(y,(2,2)))
    #print('\n')
    #print(blur(y,(2,2)))
    
    src = cv2.imread("./pics/LenaRGB.bmp")
    dst = blur(src,(9,9))
    cv2.imwrite(f"./pics/blur/blur_avg.jpg",dst)
    dst = cv2.blur(src,(9,9))
    cv2.imwrite(f"./pics/blur/blur_avg_cv.jpg",dst)    