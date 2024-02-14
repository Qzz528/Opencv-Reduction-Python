# -*- coding: utf-8 -*-
"""
add
对图像进行累加，并且避免像素超过255
dst = cv2.add(src1,src2)
src:图像矩阵（支持多通道）
ksize:双元素tuple，(kh,kw)，分别为滤波器核的高和宽
RETURN
dst:填充边缘后的图像矩阵
"""

import numpy as np
import cv2
from checktools import MaxError


def add_numpy(src1,src2): 
    assert src1.shape==src2.shape
    src1=src1.astype(float)
    src2=src2.astype(float)
    dst = src1+src2
    
    dst = np.clip(dst,0,255) 
    #或
    #dst[dst>255]=255
    return dst.astype('uint8')


def add(src1,src2):
    isimg = [type(src1)==np.ndarray, type(src2)==np.ndarray]
    assert True in isimg #保证有不全是scaler
    if not False in isimg:#如果没有scaler
        assert src1.shape==src2.shape
        dtype = src1.dtype
        src1 = src1.astype(float)
        src2 = src2.astype(float)
    else:#有scaler
        if isimg[0]:#如果src1是img
            dtype = src1.dtype
            src1 = src1.astype(float)
            src2 = src2*np.ones_like(src1)
        else:
            dtype = src2.dtype
            src2 = src2.astype(float)
            src1 = src1*np.ones_like(src2)            
    if src1.ndim==3:        
        h,w,c = src1.shape  
    else:
        h,w,c = src1.shape[0],src1.shape[1],0
    
    dst = np.zeros_like(src1)

    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    dst[i,j,k] = max(min(src1[i,j,k]+src2[i,j,k],255),0)
            else:
                 dst[i,j] = max(min(src1[i,j]+src2[i,j],255),0)
    return dst.astype(dtype)

def addWeighted_numpy(src1,w1,src2,w2,b):
    assert type(src1)==np.ndarray
    assert type(src2)==np.ndarray
    assert src1.shape==src2.shape
    src1=src1.astype(int)
    src2=src2.astype(int)
    dst = src1*w1+src2*w2+b
    
    dst = np.clip(dst,0,255)
    #或
    dst[dst>255] = 255
    dst[dst<0] = 0
    return dst.astype('uint8')

def addWeighted(src1,w1,src2,w2,b):
    assert src1.shape==src2.shape
    dst = np.zeros_like(src1).astype('uint8')
    if len(src1.shape)==3:
        h,w,c = src1.shape
    else:
        h,w,c = src1.shape[0],src1.shape[1],0
    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    val = w1*int(src1[i,j,k])+w2*int(src2[i,j,k])+b
                    dst[i,j,k] = max(min(val,255),0)
            else:
                val = w1*int(src1[i,j])+w2*int(src2[i,j])+b
                dst[i,j] = max(min(val,255),0)
    return dst


if __name__ == '__main__':
    #3通道图
    src1 = cv2.imread("./pics/LenaRGB.bmp")
    src2 = cv2.imread("./pics/sailboat.bmp")
    #单通道灰度图
    src1_gray = cv2.cvtColor(src1,cv2.COLOR_BGR2GRAY)
    src2_gray = cv2.cvtColor(src2,cv2.COLOR_BGR2GRAY)

    
    print('#'*10,'check function: add','#'*10)
    for img in [100,src2]:
        add_cv = cv2.add(src1,img)
        add_ = add(src1,img)
        print('type:',type(img), '|',
              'ndims:',src1.ndim,'|',
              'max_error:', MaxError(add_cv,add_))
    for img in [100,src2_gray]:
        add_cv = cv2.add(src1_gray,img)
        add_ = add(src1_gray,img)
        print('type:',type(img), '|',
              'ndims:',src1_gray.ndim,'|',
              'max_error:', MaxError(add_cv,add_))
    print('#'*40)

    #y = 2*np.arange(1,10).reshape(3,3).astype('uint8')
    #print(cv2.addWeighted(x,0.5,y,0.5,0))
    #print(addWeighted_numpy(x,0.5,y,0.5,0))
    #print(addWeighted_numpy(x,0.5,y,0.5,0))
    
    
    src1 = cv2.imread("./pics/LenaRGB.bmp")
    

    dst = src1+src2
    cv2.imwrite(f"./pics/add/add_raw.jpg",dst)
    dst = cv2.add(src1,src2)
    cv2.imwrite(f"./pics/add/add_cv.jpg",dst)  
    dst = add(src1,src2)
    cv2.imwrite(f"./pics/add/add.jpg",dst)   

    #dst=0.7*src1+0.3*src2-10
    #cv2.imwrite(f"./pics/add/addWeighted_raw.jpg",dst)
    #dst=cv2.addWeighted(src1,0.7,src2,0.3,-10)
    #cv2.imwrite(f"./pics/add/addWeighted_cv.jpg",dst)
    #dst=addWeighted(src1,0.7,src2,0.3,-10)
    #cv2.imwrite(f"./pics/add/addWeighted.jpg",dst)
