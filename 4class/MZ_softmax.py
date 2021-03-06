import math
import pandas as pd
import numpy as np
import random
import time
import h5py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util import *

import matplotlib.pyplot as plt


class Softmax(object):

    def __init__(self):
        self.learning_step = 0.0001           # 学习速率
        self.max_iteration = 300000         # 最大迭代次数
        self.weight_lambda = 0.01               # 衰退权重

    def cal_e(self,x,l):
        '''
        计算博客中的公式3
        '''

        theta_l = self.w[l]
        product = np.dot(theta_l,x)

        return math.exp(product)

    def cal_probability(self,x,j):
        '''
        计算博客中的公式2
        '''

        molecule = self.cal_e(x,j)
        denominator = sum([self.cal_e(x,i) for i in range(self.k)])

        return molecule/denominator


    def cal_partial_derivative(self,x,y,j):
        '''
        计算博客中的公式1
        '''

        first = int(y==j)                           # 计算示性函数
        second = self.cal_probability(x,j)          # 计算后面那个概率

        return -x*(first-second) + self.weight_lambda*self.w[j]

    def predict_(self, x):
        result = np.dot(self.w,x)
        row, column = result.shape
        
        # print(result)
        # 找最大值所在的列
        _positon = np.argmax(result)
        m, n = divmod(_positon, column)             # 除法，返回除数和余数

        return m

    def train(self, features, labels):
        self.k = len(set(labels))                   # set(labels)集合，将labels依次加入集合中
        self.w = np.zeros((self.k,len(features[0])+1))      # 初始化权重矩阵为0
        time = 0
        
        train_predict = []
        score = []
        
        while time < self.max_iteration:
            
            index = random.randint(0, len(labels) - 1)      # 每次随机一个index，即随机取出一个样本进行梯度下降，应该为 SGD

            x = features[index]
            y = labels[index]
    
            x = list(x)                         # 将元组转换为列表
            x.append(1.0)
            x = np.array(x)

            derivatives = [self.cal_partial_derivative(x,y,j) for j in range(self.k)]

            for j in range(self.k):
                self.w[j] -= self.learning_step * derivatives[j]

            if time % 10000 == 0:
                pre = p.predict(features)
                sco = accuracy_score(labels, pre)
                print('loop %d' % time)
                print("after %d loop ,the train acc score is :%f" % (time, sco))
            time += 1
            
            train_predict.append(pre)
            score.append(pre)
            
            
            
        print("the weight is :" + str(self.w))
        
        # 可视化画出3种不同类别学习到的权重值
        print(self.w[0, :-1].shape)
        plt.subplot(1,3,1)
        number = 0
        plt.imshow(self.w[number, :-1].reshape([100, 100]), cmap='gray')
        plt.hold
        plt.subplot(1, 3, 2)
        number = 1
        plt.imshow(self.w[number, :-1].reshape([100, 100]), cmap='gray')
        plt.hold
        plt.subplot(1, 3, 3)
        number = 2
        plt.imshow(self.w[ number, :-1].reshape([100, 100]), cmap='gray')
        plt.hold
        
        plt.show()
        
    def predict(self,features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)

            x = np.matrix(x)
            x = np.transpose(x)

            labels.append(self.predict_(x))
        return labels


def load_data(train_filename,test_filename):

    ########################## 从测试集中分离10%作为验证集
    train_dataset = h5py.File(train_filename, "r")
    x,dim,y,z = train_dataset["train_set_x"].shape
    train = int(x * 0.9)
    test = x - train
    
    train_set_x_orig = np.array(np.concatenate((train_dataset["train_set_x"][:900],train_dataset["train_set_x"][1000:1900],
                                                train_dataset["train_set_x"][2000:2900]),axis = 0))
    train_set_y_orig = np.array(np.concatenate((train_dataset["train_set_y"][:900],train_dataset["train_set_y"][1000:1900],
                                                train_dataset["train_set_y"][2000:2900]),axis = 0))
    test_dataset = h5py.File(test_filename, "r")
    test_set_x_orig = np.array(np.concatenate((test_dataset["train_set_x"][900:1000],test_dataset["train_set_x"][1900:2000],
                                               test_dataset["train_set_x"][2900:]),axis = 0))  # your test set features
    test_set_y_orig = np.array(np.concatenate((test_dataset["train_set_y"][900:1000],test_dataset["train_set_y"][1900:2000],
                                               test_dataset["train_set_y"][2900:]),axis = 0)) # your test set labels
    ########################

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig



if __name__ == '__main__':
    
    
    train_filename = "MultiLabel_data_right.h5"
    ##################  读取数据  #######################
    print('Start read data')
    time_1 = time.time()
    load_data(train_filename, train_filename)

    train_features, train_labels, test_features, test_labels = load_data(train_filename, train_filename)
    # 保持shape[0]不变，其他按适应性变化，即保持样本个数不变
    train_x_flatten = train_features.reshape(train_features.shape[0],-1)  # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_features.reshape(test_features.shape[0], -1)
    train_y_flatten = train_labels.reshape(train_labels.shape[0],-1)[0].tolist()  # The "-1" makes reshape flatten the remaining dimensions
    test_y_flatten = test_labels.reshape(test_labels.shape[0], -1)[0].tolist()

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.

    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    # print("train_y's shape: " + str(train_y_flatten.shape))
    # print("test_y's shape: " + str(test_y_flatten.shape))
    
    time_2 = time.time()
    print('read data cost '+ str(time_2 - time_1)+' second')



    ##################  开始训练  #######################
    print('Start training')
    p = Softmax()
    p.train(train_x, train_y_flatten)

    time_3 = time.time()
    print('training cost '+ str(time_3 - time_2)+' second')
    
    
    
    ##################  开始测试  #######################
    print('Start predicting')
    print(test_x.shape)
    test_predict = p.predict(test_x)
    
    time_4 = time.time()
    print('predicting cost ' + str(time_4 - time_3) +' second')



    ##################  结果展示  #######################
    score = accuracy_score(test_y_flatten, test_predict)
    print(test_y_flatten)
    print(test_predict)
    print("The accruacy socre is " + str(score))

    print_mislabeled_images(test_x, test_y_flatten, test_predict)       # 显示错误分类的图像