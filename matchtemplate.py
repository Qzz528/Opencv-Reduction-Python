# -*- coding: utf-8 -*-
"""
matchTemplate

result = cv2.matchTemplate(image,templ,method)

image:所要进行检测的单通道(图像)矩阵，array，尺寸为(H,W)
templ:所要进行匹配的模板，array，尺寸为(h,w)
method:image与templ进行计算的方法，int 0~5
    0 cv2.TM_SQDIFF        $R(x,y)=\sum_{x',y'}(T(x',y')−I(x+x',y+y'))^2$
    1 cv2.TM_SQDIFF_NORMED $R(x,y)=\frac {\sum_{x',y'}(T(x',y')−I(x+x',y+y'))^2}{\sqrt{\sum_{x',y'}T(x',y')^2\cdot\sum_{x',y'}I(x+x',y+y')^2}}$
    2 cv2.TM_CCORR         $R(x,y)=\sum_{x',y'}(T(x',y')\cdot I(x+x',y+y'))$
    3 cv2.TM_CCORR_NORMED  $R(x,y)=\frac {\sum_{x',y'}(T(x',y')\cdot I(x+x',y+y'))}{\sqrt{\sum_{x',y'}T(x',y')^2\cdot\sum_{x',y'}I(x+x',y+y')^2}}$
    4 cv2.TM_CCOEFF        $R(x,y)=\sum_{x',y'}(T'(x',y')\cdot I'(x+x',y+y'))$ where $T'(x',y')=T(x',y')−1/(w⋅h)⋅∑{x'',y''}T(x'',y'')$ $I'(x+x',y+y')=I(x+x',y+y')−1/(w⋅h)⋅∑{x'',y''}I(x+x'',y+y'')$
    5 cv2.TM_CCOEFF_NORMED $R(x,y)=\frac {\sum_{x',y'}(T'(x',y')\cdot I'(x+x',y+y'))}{\sqrt{\sum_{x',y'}T'(x',y')^2\cdot\sum_{x',y'}I'(x+x',y+y')^2}}$
RETURN
result:image中各位置其内容与templ的对应程度，array，尺寸为(H-h+1,W-w+1)

method为0,1时计算的是image和templ之间的差，值越小越匹配，2~5时计算的是相关性，值越大越匹配
所要检测的图像image是按滑窗平移，按行列逐个的截取其一部分与templ进行计算，值存入result对应位置
"""

import cv2
import numpy as np


def matchTemplate(image,templ,method):

    #定义匹配计算方法
    if method == 0:
        def func(a,b): #SqDiff
            return np.sum((a-b)**2)
    elif method == 1:
        def func(a,b): #SqDiff_Norm
            return np.sum((a-b)**2)/((np.sum(a**2)*np.sum(b**2))**0.5)
    elif method == 2:
        def func(a,b): #Ccor
            return np.sum(a*b)
    elif method == 3:
        def func(a,b): #Ccor_Norm
            return np.sum(a*b)/((np.sum(a**2)*np.sum(b**2))**0.5)
    elif method == 4:
        def func(a,b):  #Ccoef   
            return np.sum((a-np.mean(a))*(b-np.mean(b)))
    elif method == 5:
        def func(a,b): #Ccoef_Norm
            a_ = a-np.mean(a)
            b_ = b-np.mean(b)
            return np.sum(a_*b_)/((np.sum(a_**2)*np.sum(b_**2))**0.5) if np.sum(a_*b_)!=0 else 0                    

    #创建存放结果的array
    h,w = templ.shape
    result_h = image.shape[0] - templ.shape[0] + 1
    result_w = image.shape[1] - templ.shape[1] + 1
    result = np.zeros((result_h,result_w))

    #根据结果计算公式，遍历计算
    for y in range(result_h):
        for x in range(result_w):
            result[y,x] = func(templ,image[y:y+h,x:x+w])
    return result


if __name__ == '__main__':
    templ = np.arange(1,5).reshape(2,2)
    image = np.pad(templ, 1 ,mode='edge')
    image = image.astype('uint8')
    templ = templ.astype('uint8')
    print(image)
    print(templ)
    print('\n')
    ##根据示例数据可发现：templ与image的中心匹配度最高
    
    for i in range(6):
        print(matchTemplate(image,templ,method=i)) 
        print(cv2.matchTemplate(image,templ,method=i))
        print('\n')