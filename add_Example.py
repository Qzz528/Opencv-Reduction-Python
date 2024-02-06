# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 09:04:02 2024

@author: Administrator
"""
import cv2
img = cv2.imread("F:\LenaRGB.bmp")

print(img.shape)

blur_u = cv2.blur(img[:255,:,:],(11,11))
img_u = img.copy()
img_u[:255,:,:]=blur_u
cv2.imwrite("F:\LenaRGB_U.bmp",img_u)


blur_d = cv2.blur(img[255:,:,:],(11,11))
img_d = img.copy()
img_d[255:,:,:]=blur_d
cv2.imwrite("F:\LenaRGB_D.bmp",img_d)

RE = cv2.addWeighted(img_d,0.5,img_u,0.5,0)
cv2.imwrite("F:\LenaRGB_Re.bmp",RE)

#%%
def boxBlur(src, box_coord, ksize=(11,11)): #box_cord x1y1x2y2 None表示边界
    dst = src.copy()
    x1,y1,x2,y2=box_coord
    box = cv2.blur(src[y1:y2,x1:x2,:],ksize)
    dst[y1:y2,x1:x2,:] = box
    return dst

img_u = boxBlur(img,[0,0,None,255])
img_d = boxBlur(img,[0,255,None,None])
cv2.imwrite("F:\LenaRGB_BlurU.bmp",img_u)
cv2.imwrite("F:\LenaRGB_BlurD.bmp",img_d)

img_re = cv2.addWeighted(img_d,0.5,img_u,0.5,0)
cv2.imwrite("F:\LenaRGB_BlurRe.bmp",img_re)

def boxMask(src, box_coord, value=0): #box_cord x1y1x2y2 None表示边界
    x1,y1,x2,y2=box_coord
    dst = src.copy()
    dst[y1:y2,x1:x2,:] = value
    return dst

img_u = boxMask(img,[0,0,None,255])
img_d = boxMask(img,[0,255,None,None])
cv2.imwrite("F:\LenaRGB_MaskU.bmp",img_u)
cv2.imwrite("F:\LenaRGB_MaskD.bmp",img_d)

img_re = cv2.addWeighted(img_d,0.5,img_u,0.5,0)
cv2.imwrite("F:\LenaRGB_MaskRe.bmp",img_re)   
#直接拼接，黑色区域没有信息 不应该占据权重 

import numpy as np
def randomNoise(src, ratio, value):
    dst = src.copy()
    h,w = src.shape[0],src.shape[1]
    index = np.arange(h*w)    
    np.random.shuffle(index)
    noise_index = index[:int(h*w*ratio)]
    noise_h = noise_index//h
    noise_w = noise_index%h
    dst[noise_h,noise_w] = value
    return dst

img_u = randomNoise(img,0.1,0)
img_d = randomNoise(img,0.1,0)
cv2.imwrite("F:\LenaRGB_Noise0.bmp",img_u)
cv2.imwrite("F:\LenaRGB_Noise255.bmp",img_d)

img_re = cv2.addWeighted(img_d,0.5,img_u,0.5,0)
cv2.imwrite("F:\LenaRGB_NoiseRe.bmp",img_re)   

#%%
def addAverange(imgs):
    n_imgs = len(imgs)
    global int_imgs
    int_imgs = [_.astype(int) for _ in imgs]
    avg_img = np.mean(np.stack(int_imgs),axis=0)
    return avg_img.astype('uint8')
    

imgs = []
n_imgs = 8
for i in range(n_imgs):
    if i<n_imgs//4:
        img_noise = randomNoise(img,0.1,0)
    elif i<n_imgs//2:
        img_noise = randomNoise(img,0.1,255)
    elif i<3*n_imgs//4:
        img_noise = randomNoise(img,0.05,255)
        img_noise = randomNoise(img_noise,0.05,0)
    else:
        img_noise = randomNoise(img,0.05,0)
        img_noise = randomNoise(img_noise,0.05,255)
    
    if i%(n_imgs//4)==0:
        cv2.imwrite(f"F:\LenaRGB_Noise-{i//(n_imgs//4)}.bmp",img_noise)
    imgs.append(img_noise)
    
img_avg = addAverange(imgs)
cv2.imwrite(f"F:\LenaRGB_Noise-AVG-{n_imgs}.bmp",img_avg)
#random choise = index random & select from start


#%%
def imgUnion(src1,src2):
    assert src1.shape==src2.shape
    dst = np.zeros_like(src1)
    h,w,c = src1.shape if len(src1.shape)==3 else src1.shape[0],src1.shape[1],0
    for i in range(h):
        for j in range(w):
            if c:
                for k in range(c):
                    dst[i,j,k] = max(src1[i,j,k],src2[i,j,k])
            else:
                dst[i,j]=max(src1[i,j],src2[i,j])
    return dst

def imgUnion_numpy(src1,src2):
    assert src1.shape==src2.shape
    shape = src1.shape
    stack = np.stack((src1.reshape(-1),src2.reshape(-1)))
    dst = np.max(stack,axis=0)
    dst = ds    
#%%
aUB
for
zipmax
特别的如果src单通道且常值 直接
mean*3
rgb2gray