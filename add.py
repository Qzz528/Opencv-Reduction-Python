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


def add_numpy(src1,src2): 
    assert src1.shape==src2.shape
    src1=src1.astype(int)
    src2=src2.astype(int)
    return np.clip(src1+src2,0,255)

def add_numpy2(src1,src2):
    assert src1.shape==src2.shape
    src1=src1.astype(int)
    src2=src2.astype(int)
    dst = src1+src2
    dst[dst>255]=255
    return dst

def add(src1,src2):
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
                    dst[i,j,k] = min(int(src1[i,j,k])+int(src2[i,j,k]),255)
            else:
                 dst[i,j] = min(int(src1[i,j])+int(src2[i,j]),255)
    return dst



if __name__ == '__main__':
    
    x = 20*np.arange(1,10).reshape(3,3).astype('uint8')
    #y = np.dstack((x,x,x))
    print(cv2.add(x,x))
    print(add_numpy(x,x))
    print(add_numpy2(x,x))
    print(add(x,x))
    
    #print(cv2.blur(y,(2,2)))
    #print('\n')
    #print(blur(y,(2,2)))
    
    #src = cv2.imread("./pics/LenaRGB.bmp")
    #dst = blur(src,(9,9))
    #cv2.imwrite(f"./pics/blur/blur_avg.jpg",dst)
    #dst = cv2.blur(src,(9,9))
    #cv2.imwrite(f"./pics/blur/blur_avg_cv.jpg",dst)    

    #raw add

    #降噪

    #两者融合去模糊

    #两图叠加