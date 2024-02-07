# -*- coding: utf-8 -*-
import numpy as np
import cv2
from copyMakeBorder import copyMakeBorder_numpy, copyMakeBorder


"dst = cv2.blur(src,ksize)"
#对图像进行均值滤波
#src:图像矩阵（支持多通道）
#ksize:双元素tuple，(kh,kw)，分别为滤波器核的高和宽
#RETURN
#dst:滤波后图像（与src同尺寸）

#自写方法
def blur(src,ksize,borderType=4,anchor=(-1,-1),value=0):
    kw,kh = ksize
    h,w = src.shape[:2]
    c = src.shape[2] if len(src.shape)==3 else 0
    #四周填充的行列数取决于anchor，默认为(-1,-1)，即卷积核中心为锚点，四周均匀填充
    if anchor[0]==anchor[1]==-1:    
        top = kh//2
        left = kw//2
    else:
        top = anchor[1]
        left = anchor[0]
    bottom = kh-1-top
    right = kw-1-left
    #行列总填充分别为kh-1，kw-1
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


"dst = cv2.medianBlur(src,ksize)"
#对图像进行中值滤波
#src:图像矩阵（支持多通道）
#ksize:int, 滤波器核的高和宽值（两者相同）
#RETURN
#dst:滤波后图像（与src同尺寸）

#自写方法
def medianBlur(src,ksize,borderType=4,value=0): 
    kh,kw = ksize,ksize #ksize只接受整数，核高宽一致
    assert ksize%2==1
    h,w = src.shape[:2]

    c = src.shape[2] if len(src.shape)==3 else 0
    #四周填充
    top = kh//2
    left = kw//2
    bottom = kh-1-top
    right = kw-1-left
    #行列总填充分别为kh-1，kw-1

    #先扩充src确保运算后dst大小与原src一致
    pad_src = copyMakeBorder(src,top,bottom,left,right,borderType,value)
    
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
    #由于float到int的舍入，有时自写方法会和opencv的会相差1

    src = cv2.imread("./pics/LenaRGB.bmp")
    src = cv2.resize(src,(64,64))


    print('#'*10,'check function: blur','#'*10)
    for ksize in [(8,8),(9,9),(8,9),(9,8)]:
        for borderType in [0,1,2,4]:
            for anchor in [(-1,-1),(3,3),(4,4),(3,4),(4,3)]:
                dst_cv = cv2.blur(src,ksize,borderType=borderType,anchor=anchor)
                dst = blur(src,ksize,borderType,anchor)
                print('ksize:',ksize, '|',
                      'borderType:',borderType,'|',
                      'anchor:',anchor,'|',
                      'max_error:', np.max(np.abs(dst_cv-dst)))
    print('#'*40)

    print('#'*10,'check function: medianBlur','#'*10)
    for ksize in [5,9,15]:
        dst_cv = cv2.medianBlur(src,ksize)
        dst = medianBlur(src,ksize,borderType=1)
        print('ksize:',ksize, '|',
              'max_error:', np.max(np.abs(dst_cv-dst)))
    print('#'*40)

    #src = cv2.imread("./pics/LenaRGB.bmp")
    #dst = blur(src,(9,9))
    #cv2.imwrite(f"./pics/blur/blur_avg.jpg",dst)
    #dst = cv2.blur(src,(9,9))
    #cv2.imwrite(f"./pics/blur/blur_avg_cv.jpg",dst)    