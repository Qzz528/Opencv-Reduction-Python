# -*- coding: utf-8 -*-
"""
medianBlur

dst = cv2.medianBlur(src,ksize)
"""

import numpy as np
import cv2
from opencv_copyMakeBorder import makeborder_numpy, makeborder



def blur_median(src,ksize,borderType=2,value=0,srd = True): #**kwargs
    kh,kw = ksize,ksize #ksize只接受整数，核高宽一致
    h,w = src.shape[:2]

    c = src.shape[2] if len(src.shape)==3 else 0


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
    pad_src = makeborder(src,top,bottom,left,right,borderType,value)
    
    dst = np.zeros_like(src)
    
    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    dst[i,j,k] = np.median(pad_src[i:i+kh,j:j+kw,k])
                    #如果不使用np.median求中位数需要排序
                    # seq = list(pad_src[i:i+kh,j:j+kw,k].reshape(-1))
                    # seq.sort()
                    # dst[i,j,k] = seq[len(seq)//2]
            else:
                dst[i,j] = np.median(pad_src[i:i+kh,j:j+kw])
                #如果不使用np.median求中位数需要排序
                # seq = list(pad_src[i:i+kh,j:j+kw].reshape(-1))
                # seq.sort()
                # dst[i,j] = seq[len(seq)//2]

    return dst



if __name__ == '__main__':
    
    x = np.arange(1,10).reshape(3,3).astype('uint8') #(3,3)
    y = np.dstack((x,x,x)) #(3,3,3channel)
    

    print(cv2.medianBlur(x,3))
    print(blur_median(x,3))
    print('\n')
    

    print(cv2.medianBlur(y,3))
    print(blur_median(y,3))    
    