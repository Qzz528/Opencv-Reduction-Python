# -*- coding: utf-8 -*-
"""
blur

dst = cv2.blur(src,ksize)
"""

import numpy as np
import cv2
from opencv_copyMakeBorder import makeborder_numpy, makeborder



def blur_avg(src,ksize,borderType=4,value=0,srd = False): #**kwargs
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
    else: #opencv使用的填充方式是左上方填充
        top = kh-1
        left = kw-1
        bottom = 0
        right = 0
    
    #先扩充src确保运算后dst大小与原src一致
    pad_src = makeborder(src,top,bottom,left,right,borderType,value)
    
    
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
    y = np.dstack((x,x,x))
    

    
    print(cv2.blur(x,(2,2)))
    print(blur_avg(x,(2,2)))
    print('\n')

    print(cv2.blur(x,(2,2),borderType=1))
    print(blur_avg(x,(2,2),borderType=1))
    print('\n')

    
    print(cv2.blur(y,(2,2)))
    print('\n')
    print(blur_avg(y,(2,2)))
    
    