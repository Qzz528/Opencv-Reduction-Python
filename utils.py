# -*- coding: utf-8 -*-

import numpy as np
import cv2

#计算自写方法和opencv方法的最大像素差
def MaxError(dst1, dst2):
    dst1 = dst1.astype(float)
    dst2 = dst2.astype(float)
    err = np.abs(dst1-dst2)
    return err.max()


'dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType, value)'
#滤波计算前的准备，对图像边缘进行填充

#src:图像矩阵（支持多通道）
#top,bottom,left,right:顶部，底部，左部，右部边缘要填充的像素值
#borderType:边缘填充方式，int 0~4 [value为常数填充时填充的数值]
#    0 cv2.BORDER_CONSTANT  常数填充，如123->0012300  (value=0)
#    1 cv2.BORDER_REPLICATE 复制填充，如123->1112333
#    2 cv2.BORDER_REFLECT   反射填充，如123->2112332
#    3 cv2.BORDER_WRAP      循环填充，如123->2312312
#    4 cv2.BORDER_DEFAULT   默认填充（不重复的反射），如123->3212321
#RETURN
#dst:填充边缘后的图像矩阵

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
    dst[top:-bottom if bottom else None,left:-right if right else None] = src
        
    if borderType==0:
        return dst.astype('uint8')
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

    return dst.astype('uint8')
        
'kernel=cv2.getGaussianKernel(ksize,sigma)'
#获取一维的高斯核
#ksize:高斯核尺寸
#sigma:高斯分布标准差
#RETURN
#kernel:数据尺寸为(ksize,1), 且元素和为1的array
def getGaussianKernel(ksize, sigma):
    assert ksize>0 and ksize%2==1
    center = (ksize-1)/2
    x = np.arange(ksize)-center #平移，保证中心元素x为0
    #x根据高斯密度求kernel
    kernel = np.exp(-(x**2)/(2*(sigma**2))) #最终要归一化，可省略系数
    kernel = kernel/np.sum(kernel) #归一化
    return kernel[:,None] #加一维度，返回形状(ksize,1)


#ksize:双元素tuple, 高斯核尺寸宽和高
#sigma:双元素tuple, 高斯分布标准差, 横向和纵向
def getGaussianKernel2D(ksize, sigma):#根据1维核获取2维
    kernelX = getGaussianKernel(ksize[0],sigma[0])
    kernelY = getGaussianKernel(ksize[1],sigma[1]) 
    kernel = kernelY @ kernelX.T #kernelX和kernelY已经归一化，矩阵乘和仍是1  
    return kernel
def getGaussianKernel2D_(ksize, sigma):#直接根据公式生成2维核
    centerX = (ksize[0]-1)/2
    x = np.arange(ksize[0])-centerX #平移，保证中心元素为0
    centerY = (ksize[1]-1)/2
    y = np.arange(ksize[1])-centerY #平移，保证中心元素为0
    kernel = np.zeros((ksize[1],ksize[0]))
    for i in range(ksize[1]):
        for j in range(ksize[0]):
            kernel[i,j] = np.exp(-(x[j]**2+y[i]**2)/(2*sigma[0]*sigma[1]))
    #归一化
    kernel = kernel/kernel.sum()
    
    return kernel


'dst=cv2.filter2D(src,ddepth,kernel)'
#使用指定的2维核kernel对src图像进行滤波
#src:图像矩阵（支持多通道）
#kernel:2维的滤波器核
#ddepth取-1
###borderType: 可选参数, int, 边缘填充方式, 详见copyMakeBorder
###anchor: 可选参数，双元素tuple, 滤波器核的中心坐标, 决定了copyMakeBorder四周填充的长度

def filter2D(src,kernel,borderType=4,anchor=(-1,-1),value=0):
    kh,kw = kernel.shape[:2]
    h,w = src.shape[:2]
    c = src.shape[2] if len(src.shape)==3 else 0
    #四周填充的行列数取决于anchor，
    if anchor[0]==anchor[1]==-1:#默认为(-1,-1)，即卷积核中心为锚点，四周均匀填充
        top = kh//2
        left = kw//2
    else:#否则根据anchor值填充
        top = anchor[1]
        left = anchor[0]
    bottom = kh-1-top
    right = kw-1-left
    #行列总填充分别为kh-1，kw-1
    #先扩充src确保运算后dst大小与原src一致
    pad_src = copyMakeBorder(src,top,bottom,left,right,borderType,value)
    
    dst = np.zeros_like(src).astype(float)
    #逐行逐列的进行卷积
    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    dst[i,j,k] = round(np.sum(pad_src[i:i+kh,j:j+kw,k]*kernel))
            else:
                dst[i,j] = round(np.sum(pad_src[i:i+kh,j:j+kw]*kernel))
    return dst

if __name__ == '__main__':

    src = cv2.imread("./pics/LenaRGB.bmp")
    src = cv2.resize(src,(64,64))

    print('#'*10,'check function: copyMakeBorder','#'*10)
    for top in [0,1,2]:
        for bottom in [0,1,2]:
            for left in [0,1,2]:
                for right in [0,1,2]:
                    for borderType in range(5):#value!=0 时opencv方法只对第一通道填充指定值
                        dst_cv = cv2.copyMakeBorder(src, top, bottom, left, right, borderType =borderType)
                        dst_np = copyMakeBorder_numpy(src,top,bottom,left,right,borderType)
                        dst = copyMakeBorder(src,top,bottom,left,right,borderType)
                        print('top,bottom,left,right:',top,bottom,left,right, '|',
                              'borderType:',borderType,'|',
                              'max_error:', MaxError(dst_cv,dst), MaxError(dst_cv,dst_np))
    print('#'*40)


    print('#'*10,'check function: getGaussianKernel','#'*10)
    for ksize in [1,3,5]:
        for sigma in [0.1,1,10]:
            k_cv = cv2.getGaussianKernel(ksize,sigma)
            k = getGaussianKernel(ksize,sigma)
            print('ksize:',ksize, '|',
                  'sigma:',sigma, '|',
                  'max_error:', MaxError(k_cv,k))
    print('#'*40)

    print('#'*10,'check function: getGaussianKernel2D','#'*10)
    for ksize in [(3,3),(5,5),(3,5),(5,3)]:
        for sigma in [(1,1),(10,10),(1,10),(10,1)]:
            k = getGaussianKernel2D(ksize,sigma)
            k_ = getGaussianKernel2D_(ksize,sigma)
            print('ksize:',ksize, '|',
                  'sigma:',sigma, '|',
                  'max_error:', MaxError(k,k_))
    print('#'*40)


    print('#'*10,'check function: filter2D','#'*10)
    print('由于舍入误差，可能与opencv像素值最多相差1')
    k1 = np.arange(20).reshape(4,5);k1=k1/k1.sum()#shape of (4,5)
    k2 = k1.T #shape of (5,4)
    k3 = np.arange(16).reshape(4,4);k3=k3/k3.sum()#shape of (4,4)
    k4 = np.arange(25).reshape(5,5);k4=k4/k4.sum()#shape of (4,4)

    for kernel in [k1,k2,k3,k4]:
        for borderType in [0,1,2,4]:
            for anchor in [(-1,-1),(1,1),(2,2),(1,2),(2,1)]:
                dst_cv = cv2.filter2D(src,-1,kernel,borderType=borderType,anchor=anchor)
                dst = filter2D(src,kernel,borderType,anchor)
            print('kernel_size:',kernel.shape, '|',
                  'borderType:',borderType, '|',
                  'anchor:',anchor, '|',
                  'max_error:', MaxError(dst_cv,dst))
    print('#'*40)    



    #src = cv2.imread("./pics/LenaRGB.bmp")
    #for i in range(5):
    #    dst = copyMakeBorder(src,32,32,32,32,i)
    #    cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}.jpg",dst)
    #    dst = copyMakeBorder_numpy(src,32,32,32,32,i)
    #    cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}_np.jpg",dst)
    #    dst = cv2.copyMakeBorder(src,32,32,32,32,i)
    #    cv2.imwrite(f"./pics/copyMakeBorder/copyMakeBorder-{i}_cv.jpg",dst)