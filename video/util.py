import math
import pandas as pd
import numpy as np
import random
import time
import h5py


def cal_e(x, l, ws):
    theta_l = ws[l]
    product = np.dot(theta_l, x)
    
    return math.exp(product)


def cal_probability(x, j, k):
    molecule = cal_e(x, j)
    denominator = sum([cal_e(x, i) for i in range(k)])
    return molecule / denominator


def cal_partial_derivative(x, y, j, ws):
    first = int(y == j)  # 计算示性函数
    second = cal_probability(x, j)  # 计算后面那个概率
    return -x * (first - second) + weight_lambda * ws[j]


def predict_(x):
    result = np.dot(w, x)
    row, column = result.shape
    
    _positon = np.argmax(result)
    m, n = divmod(_positon, column)  # 除法，返回除数和余数
    
    return m
def softmax(Z, ws):
    
    result = np.dot(ws, Z)
    row, column = reulst.shape
    
    _position = np.argmax(result)
    m, n = divimod(_position, column)
    A = m
    
    return A


def softmax_backward(features, labels, ws):
    
    k = 3                           # 总共分类数
    sum_train = len(labels)         # 样本数量
    derivatives = []                # 存储所有样本的导数
    for i in range(sum_train):      # 依次求得每个样本在softmax之后的导数
        x = features[i]
        y = labels[i]
        x = list(x)  # 将元组转换为列表
        x.append(1.0)
        x = np.array(x)
        
        derivative = [cal_partial_derivative(x, y, j, ws) for j in range(k)]
        derivatives.append(derivative)
    
    return np.sum(abs(derivatives)) / sum_train