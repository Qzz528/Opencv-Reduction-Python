# -*- coding: utf-8 -*-

import numpy as np
import cv2

#计算自写方法和opencv方法的最大像素差
def MaxError(dst1, dst2):
    dst1 = dst1.astype(float)
    dst2 = dst2.astype(float)
    err = np.abs(dst1-dst2)
    return err.max()

#遍历params参数组合，求opencv方法和自写方法func的MaxError

def ParamsCheck(func1, func2, params_dict):

    params_list = list(params_dict.keys())
    n_params = [len(x) for x in params_dict.values()]
    n_total = int(np.prod(n_params))
    n_index = np.arange(n_total).reshape(*n_params)
    
    for i in range(n_total):
        params_index = np.argwhere(n_index==i)[0]
        #print(params_index)
        params_temp = {}
        for j,k in zip(params_list,params_index):
            params_temp[j] = params_dict[j][k]
        dst1 = func1(**params_temp)
        dst2 = func2(**params_temp)
        output_dict = {}
        for k,v in params_temp.items():
            if type(v) ==np.ndarray:
                output_dict[k+'_shape']=v.shape
            else:
                output_dict[k] = v
        #output_dict={k:v if type(v) !=np.ndarray else 'shape'+str(v.shape) for k,v in params_temp.items() }
        print(
            output_dict,
            '|error:',
            MaxError(dst1,dst2))

    return

'''
def ParamsCheck(func1, func2, params_dict, fixed_params_dict):

    params_list = list(params_dict.keys())
    #n_params = [len(x) if type(x) != np.array else 1 for x in params_dict.values()]
    n_params = [len(x) for x in params_dict.values()]
    n_total = int(np.prod(n_params))
    #print(n_total)
    
    fixed_params_shape={i:j.shape if type(j)==np.ndarray else type(j) for i,j in fixed_params_dict.items()}
    if len(params_dict):
        n_index = np.arange(n_total).reshape(*n_params)
    
        for i in range(n_total):
            params_index = np.argwhere(n_index==i)[0]
            #print(params_index)
            params_temp = {}
            for j,k in zip(params_list,params_index):
                params_temp[j] = params_dict[j][k]
            dst1 = func1(**fixed_params_dict,**params_temp)
            dst2 = func2(**fixed_params_dict,**params_temp)
            #output_dict={k:v for k,v in params_temp.items() if type(v) !=np.ndarray}
            print(
                fixed_params_shape,
                params_temp,
                '|error:',
                MaxError(dst1,dst2))
    else:
        dst1 = func1(**fixed_params_dict)
        dst2 = func2(**fixed_params_dict)
        print(
                fixed_params_shape,
                '|error:',
                MaxError(dst1,dst2))
    return
'''