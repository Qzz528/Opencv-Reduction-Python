# -*- coding: utf-8 -*-
"""
copyMakeBorder

滤波计算前的必备，对图像边缘进行填充
dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)

src:图像矩阵（支持多通道）
top,bottom,left,right:顶部，底部，左部，右部边缘要填充的像素值
borderType:边缘填充方式，int 0~4 [value为常数填充时填充的数值]
    0 cv2.BORDER_CONSTANT  常数填充，如123->0012300  (value=0)
    1 cv2.BORDER_REPLICATE 复制填充，如123->1112333
    2 cv2.BORDER_REFLECT   反射填充，如123->2112332
    3 cv2.BORDER_WRAP      循环填充，如123->2312312
    4 cv2.BORDER_DEFAULT   默认填充（不重复的反射），如123->3212321
RETURN
dst:填充边缘后的图像矩阵
"""
import numpy as np
import cv2

#利用numpy方法复现
def copyMakeBorder_numpy(src,top,bottom,left,right,borderType=4,value=0):
    #注意opencv和numpy填充名不同，特别注意numpy的reflect对应的是opencv的DEFAULT
    mode = ['constant','edge','symmetric','wrap','reflect']
    
    n_dims = len(src.shape)
    if n_dims==2:
        if borderType == 0:
            return np.pad(src, ((top,bottom),(left,right)), mode[borderType], constant_values = value)
        else:
            return np.pad(src, ((top,bottom),(left,right)), mode[borderType])
    elif n_dims==3: #如果有第三个维度（RGB彩色图），第三个色彩通道不进行填充
        if borderType == 0:
            return np.pad(src, ((top,bottom),(left,right),(0,0)), mode[borderType], constant_values = value)
        else:
            return np.pad(src, ((top,bottom),(left,right),(0,0)), mode[borderType])
    else:
        return

#手动拼接复现        
def copyMakeBorder(src,top,bottom,left,right,borderType=4,value=0):
    n_dims = len(src.shape)
    h,w = src.shape[:2]
    # global dst
    if n_dims==2:
        dst = value*np.ones((h+top+bottom,w+left+right))
    elif n_dims==3:
        dst = value*np.ones((h+top+bottom,w+left+right,src.shape[2]))
  
    #当bottom或right=0时，:-bottom和:-right失效，需要变成:None来取到最后一位
    # if bottom!=0 and right!=0:
    #     dst[top:-bottom,left:-right] = src
    # if bottom!=0 and right==0:
    #     dst[top:-bottom,left:] = src
    # if bottom==0 and right!=0:
    #     dst[top:,left:-right] = src
    # if bottom==0 and right==0:
    #     dst[top:,left:] = src
        
    dst[top:-bottom if bottom else None,left:-right if right else None] = src
        
    if borderType==0:
        return dst
    #要在边缘填充的内容，其在src中的位置序号
    if borderType == 1:
        pad_index_top = [0]*top 
        pad_index_bottom = [h-1]*bottom
        pad_index_left = [0]*left
        pad_index_right = [w-1]*right
        
    elif borderType == 2:
        seq = list(np.arange(h)) + list(np.arange(h-1,-1,-1))
        pad_index_top = [seq[i%(2*h)] for i in range(top)][::-1]
        
        seq = list(np.arange(h-1,-1,-1)) + list(np.arange(h))
        pad_index_bottom = [seq[i%(2*h)] for i in range(bottom)][::-1]
        
        seq = list(np.arange(w)) + list(np.arange(w-1,-1,-1))
        pad_index_left = [seq[i%(2*w)] for i in range(left)][::-1]
        
        seq = list(np.arange(w-1,-1,-1)) + list(np.arange(w))
        pad_index_right = [seq[i%(2*w)] for i in range(right)][::-1]
        
    elif borderType==3:
        seq = list(np.arange(h-1,-1,-1))
        pad_index_top = [seq[i%h] for i in range(top)][::-1]
        
        seq = list(np.arange(h))
        pad_index_bottom = [seq[i%h] for i in range(bottom)][::-1]
        
        seq = list(np.arange(w-1,-1,-1))
        pad_index_left = [seq[i%w] for i in range(left)][::-1]
        
        seq = list(np.arange(w))
        pad_index_right = [seq[i%w] for i in range(right)][::-1]
        

    elif borderType==4:
        seq = list(np.arange(1,h)) + list(np.arange(h-2,-1,-1))
        pad_index_top = [seq[i%(2*h-2)] for i in range(top)][::-1]
        
        seq = list(np.arange(h-2,-1,-1)) + list(np.arange(1,h))
        pad_index_bottom = [seq[i%(2*h-2)] for i in range(bottom)][::-1]
        
        seq = list(np.arange(1,w)) + list(np.arange(w-2,-1,-1))
        pad_index_left = [seq[i%(2*w-2)] for i in range(left)][::-1]
        
        seq = list(np.arange(w-2,-1,-1)) + list(np.arange(1,w))
        pad_index_right = [seq[i%(2*w-2)] for i in range(right)][::-1]        
        
    #根据src对应的index行填充顶部底部
    for i,j in zip(range(top),pad_index_top):
        dst[i, left:-right if right else None] = src[j,:] 
    for i,j in zip(range(bottom),pad_index_bottom):
        dst[-i-1, left:-right if right else None] = src[j,:] 
    #按填充完dst对应的index列填充左部右部（使用dst填充可以一次性填充四角）
    #dst是src的扩充，dst对应的列index = left+src对应的列index
    for i,j in zip(range(left),pad_index_left):
        dst[:,i] = dst[:,left+j]
    for i,j in zip(range(right),pad_index_right):
        dst[:,-i-1] = dst[:,left+j]   

    return dst        
        

if __name__ == '__main__':
    
    src = np.arange(1,10).reshape(3,3)
    print(src)
    print('\n')
    
    for i in range(5):
        print(cv2.copyMakeBorder(src, 2, 2, 2, 2, i))
        print(copyMakeBorder_numpy(src,2,2,2,2,i))
        print(copyMakeBorder(src,2,2,2,2,i))
        print('\n')

    src = cv2.imread("./pics/LenaRGB.bmp")
    for i in range(5):
        dst = copyMakeBorder(src,32,32,32,32,i)
        cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}.jpg",dst)
        dst = copyMakeBorder_numpy(src,32,32,32,32,i)
        cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}_np.jpg",dst)
        dst = cv2.copyMakeBorder(src,32,32,32,32,i)
        cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}_cv.jpg",dst)