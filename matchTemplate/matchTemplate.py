import cv2
import numpy as np

'创造示例数据image与templ'
templ = np.arange(1,5).reshape(2,2)
image = np.pad(templ, 1 ,mode='edge')

image = image.astype('uint8')
templ = templ.astype('uint8')

print(image)
#[[1 1 2 2]
# [1 1 2 2]
# [3 3 4 4]
# [3 3 4 4]]
print(templ)
#[[1 2]
# [3 4]]

##根据示例数据可发现：templ与image的中心匹配度最高

'自写函数'
def matchTemplate(img,tpl,method):

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
    h,w = tpl.shape
    result_h = img.shape[0] - tpl.shape[0] + 1
    result_w = img.shape[1] - tpl.shape[1] + 1
    result = np.zeros((result_h,result_w))

    #根据结果计算公式，遍历计算
    for y in range(result_h):
        for x in range(result_w):
            result[y,x] = func(tpl,img[y:y+h,x:x+w])
    return result


'调用函数'
print(matchTemplate(image,templ,method=3)) ##自写方法
#[[0.91287093 0.92376043 0.91287093]
# [0.9797959  1.         0.98149546]
# [0.91287093 0.929516   0.91287093]]
print(cv2.matchTemplate(image,templ,method=3)) ##opencv方法
#[[0.9128709  0.92376035 0.9128709 ]
# [0.9797959  1.         0.9814954 ]
# [0.91287094 0.92951584 0.9128709 ]]