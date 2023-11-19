# -*- coding: utf-8 -*-

import cv2
import numpy as np

'创建示例，单个像素'
rgb = np.zeros((1,1,3),'uint8')
rgb[:,:,0] = 10 #R
rgb[:,:,1] = 20 #G
rgb[:,:,2] = 45 #B


'自写方法'
def rgb2hsv(rgb):#入参rgb为uint8格式

    H,W,C = rgb.shape #获取图片尺寸
    hsv = np.zeros_like(rgb,float) 

    #归一化，RGB范围0~255->0~1
    r,g,b = rgb[:,:,0]/255,rgb[:,:,1]/255,rgb[:,:,2]/255 
    
    
    #对图片的每个像素点进行RGB->HSV的转换
    for y in range(H):
        for x in range(W):
            Cmax = max(r[y,x],g[y,x],b[y,x])
            Cmin = min(r[y,x],g[y,x],b[y,x])
            hsv[y,x,2] = Cmax #求取V
            hsv[y,x,1] = (Cmax-Cmin)/Cmax if Cmax!=0 else 0 #求取S
            #求取H
            if Cmax == Cmin:
                hsv[y,x,0] = 0
            elif Cmax == r:
                if g>=b:
                    hsv[y,x,0] = 60*(g[y,x]-b[y,x])/(Cmax-Cmin) 
                else:
                    hsv[y,x,0] = 60*(g[y,x]-b[y,x])/(Cmax-Cmin) + 360
            elif Cmax == g:
                hsv[y,x,0] = 60*(b[y,x]-r[y,x])/(Cmax-Cmin) + 120
            elif Cmax == b:
                hsv[y,x,0] = 60*(r[y,x]-g[y,x])/(Cmax-Cmin) + 240

    #uint8格式，可存储的数值范围为0~255的整数，因此要对hsv值调整        
    hsv[:,:,2] = np.round(hsv[:,:,2]*255) #调整V范围0~1->0~255
    hsv[:,:,1] = np.round(hsv[:,:,1]*255) #调整S范围0~1->0~255
    hsv[:,:,0] = np.round(hsv[:,:,0]/2) #调整H范围0~360->0~180

    return hsv.astype('uint8') #输出时格式转为uint8


def hsv2rgb(hsv):#入参hsv为uint8格式
    #还原因为uint8格式而受限的hsv范围，S&V：0~255->0~1，H：0~180->0~360  
    h,s,v =  hsv[:,:,0]*2,hsv[:,:,1]/255,hsv[:,:,2]/255
    
    H,W,C = hsv.shape #获取图片尺寸
    rgb = np.zeros_like(hsv,float)
    
    #对图片的每个像素点进行HSV->RGB的转换
    for y in range(H):
        for x in range(W):
            h_index= h[y,x]//60 #根据色调判断色区 #商
            h_remain = h[y,x]/60 - h_index #余数（除60）

            #根据RGB->HSV的公式反解，求从大到小的三个值
            Cmax = v[y,x]
            Cmin = v[y,x]*(1 - s[y,x])
            Cmid_pos = v[y,x]*(1-s[y,x]+s[y,x]*h_remain)
            Cmid_neg = v[y,x]*(1-s[y,x]*h_remain)
            
            #根据H商所处的区域可得RGB三者大小关系，将其与所求的从大到小三个值对应
            if h_index == 0 : #r>g>b
                rgb[y,x,0] = Cmax
                rgb[y,x,1] = Cmid_pos
                rgb[y,x,2] = Cmin
            elif h_index == 1: #g>r>b
                rgb[y,x,1] = Cmax
                rgb[y,x,0] = Cmid_neg
                rgb[y,x,2] = Cmin
            elif h_index == 2: #g>b>r
                rgb[y,x,1] = Cmax
                rgb[y,x,2] = Cmid_pos
                rgb[y,x,0] = Cmin
            elif h_index == 3: #b>g>r
                rgb[y,x,2] = Cmax
                rgb[y,x,1] = Cmid_neg
                rgb[y,x,0] = Cmin
            elif h_index == 4: #b>r>g
                rgb[y,x,2] = Cmax
                rgb[y,x,0] = Cmid_pos
                rgb[y,x,1] = Cmin
            elif h_index == 5: #r>b>g
                rgb[y,x,0] = Cmax
                rgb[y,x,2] = Cmid_neg
                rgb[y,x,1] = Cmin

    #uint8格式，将求得的RGB范围由0~1调整到0~255
    rgb[:,:,2] = np.round(rgb[:,:,2]*255)
    rgb[:,:,1] = np.round(rgb[:,:,1]*255)
    rgb[:,:,0] = np.round(rgb[:,:,0]*255)                
        
    return rgb.astype('uint8') #输出时格式转为uint8



'调用函数'
#RGB->HSV
hsv1 = cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV) #opencv现成方法
hsv2 = rgb2hsv(rgb) #自写方法
print(hsv1,hsv2) #[[[111 198  45]]] [[[111 198  45]]]

#HSV->RGB
re_rgb1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2RGB)
re_rgb2 = hsv2rgb(hsv2)
print(re_rgb1,re_rgb2) #[[[10 21 45]]] [[[10 21 45]]]
