import math
import pandas as pd
import numpy as np
import random
import time
import h5py
import matplotlib.pyplot as plt
import csv



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


def print_mislabeled_images(X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    # 将y和p变为array格式，方便处理
    y = np.asarray(y)
    p = np.asarray(p)
    mislabeled_indices = np.asarray(np.where(y != p))  # 找到预测结果和实际结果不同的测试样本的索引
    print("mislabeled_indices.shape is : " + str(mislabeled_indices.shape))
    print("mislabeled_indices is : " + str(mislabeled_indices))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[0][i]
        # print("the pic size is: " + str(X[index, :].shape))
        # print(type(X[index, :].shape))
        
        plt.subplot(4, num_images / 4 + 1, i + 1)  # 分5行显示
        
        # 为了不同模式，图片大小不一样而改变
        # plt.imshow(X[:,index].reshape(100,100,3), interpolation='nearest')
        plt.imshow(X[index].reshape(64, 64))
        
        plt.axis('off')
        
        # 用Result函数求出每个label对应的文字结果
        pResult = Result(p[index])
        yResult = Result(y[index])
        plt.title("P: " + pResult + " \n L: " + yResult)
        
        plt.hold
        
    plt.show()

# 求出label对应的文字结果
def Result(softmaxres):
    res = ['left', 'midle', 'right', 'point']
    for i in range(softmaxres.shape[0]):
        if softmaxres[i] == 1:
            return res[i]
    
    
# 载入h5格式的数据集
def load_data(train_filename, test_filename):
    
    train_dataset = h5py.File(train_filename, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File(test_filename, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


# CSV 转矩阵
# 借助list，将csv中每一项append到list中，再进行 list-->array 转换
def CSV_2_ndarray(csvFileName):
    
    csvFile = open(csvFileName, "r")
    reader = csv.reader(csvFile)
    label = []
    data = []
    for item in reader:
        
        temp_data = np.array(item[:-1]).reshape(64, 64)
        temp_label = np.array(item[-1])
        data.append(temp_data)
        label.append(temp_label)

    label = np.array(label)
    data = np.array(data)
        
    return data, label

# 载入 csv 的数据集
def load_data_csv(train_filename, test_filename):
    
    train_set_x_orig, train_set_y_orig = CSV_2_ndarray(train_filename)
    test_set_x_orig, test_set_y_orig = CSV_2_ndarray(test_filename)
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig