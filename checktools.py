# -*- coding: utf-8 -*-

import numpy as np
import cv2

#计算自写方法和opencv方法的最大像素差
def MaxError(dst1, dst2):
    dst1 = dst1.astype(float)
    dst2 = dst2.astype(float)
    err = np.abs(dst1-dst2)
    return err.max()

#遍历params参数组合，求opencv方法func_cv和自写方法func的MaxError
def ParamsCheck(func_cv, func, params_dict):
    params_list = list(params_dict.keys())
    n_params = [len(x) for x in params_dict.values()]
    n_total = np.prod(n_params)
    n_index = np.arange(n_total).reshape(*n_params)
    for i in range(n_total):
        params_index = np.argwhere(n_index==i)
        params_temp = {}
        for j,k in zip(params_list,params_index):
            params_temp[j] = params_dict[j][k]
