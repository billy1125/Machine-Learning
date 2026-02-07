# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:13:37 2019

@author: hwdon
"""

import numpy as np
import math

def calculate_fan_in_and_fan_out(tensor):
    # 檢查 tensor 維度，至少要 2 維
    if len(tensor.shape) < 2:
        raise ValueError("tensor with fewer than 2 dimensions")

    # 若是全連接層 (2D)
    if len(tensor.shape) == 2:
        fan_in, fan_out = tensor.shape
    else:  # 卷積層權重 (F, C, kH, kW)
        num_input_fmaps = tensor.shape[1]   # 輸入通道數
        num_output_fmaps = tensor.shape[0]  # 輸出通道數
        receptive_field_size = tensor[0][0].size  # kernel 大小
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size   

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1.):
    # 計算 fan_in 與 fan_out
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    # 計算標準差
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # 轉成 uniform 分布上下界
    bound = math.sqrt(3.0) * std  
    # 使用 Xavier uniform 初始化
    tensor[:] = np.random.uniform(-bound, bound, (tensor.shape))


def xavier_normal(tensor, gain=1.):
    # 計算 fan_in 與 fan_out
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    # 計算標準差
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    # 使用 Xavier normal 初始化
    tensor[:] = np.random.normal(0, std, (tensor.shape))


# copy from Pytorch
def calculate_gain(nonlinearity, param=None):
    # 根據啟用函數回傳建議的 gain 值
    r"""Return the recommended gain value for the given nonlinearity function."""
    linear_fns = [
        'linear', 'conv1d', 'conv2d', 'conv3d',
        'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d'
    ]

    # 線性或 sigmoid
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    # tanh
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    # relu
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    # leaky relu
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False 也是 int，所以要額外判斷
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def kaiming_uniform(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    # 計算 fan_in 與 fan_out
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    # 決定使用 fan_in 或 fan_out
    if mode == 'fan_in':
        fan = fan_in
    else:
        fan = fan_out

    # 依照啟用函數計算 gain
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    # uniform 分布上下界
    bound = math.sqrt(3.0) * std      
    # 使用 Kaiming uniform 初始化
    tensor[:] = np.random.uniform(-bound, bound, (tensor.shape))


def kaiming_normal(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    # 計算 fan_in 與 fan_out
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    # 決定使用 fan_in 或 fan_out
    if mode == 'fan_in':
        fan = fan_in
    else:
        fan = fan_out

    # 依照啟用函數計算 gain
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # 從標準差推回 uniform 界線（此行僅計算）
    # 使用 Kaiming normal 初始化
    tensor[:] = np.random.normal(0, std, (tensor.shape))


def kaiming(tensor, method_params=None):
    # 預設參數設定
    method_type, a, mode, nonlinearity = 'uniform', 0, 'fan_in', 'leaky_relu'

    # 若有傳入自訂參數
    if method_params:
        method_type = method_params.get('type', "uniform")
        a = method_params.get('a', 0)
        mode = method_params.get('mode', 'fan_in')
        nonlinearity = method_params.get('nonlinearity', 'leaky_relu')

    # 根據方法選擇 uniform 或 normal
    if method_params == "uniform":
        kaiming_uniform(tensor, a, mode, nonlinearity)
    else:
        kaiming_normal(tensor, a, mode, nonlinearity)
