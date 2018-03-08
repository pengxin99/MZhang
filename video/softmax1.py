import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import h5py
 
def scale_n(x):
    return x
    #return (x-x.mean(axis=0))/(x.std(axis=0)+1e-10)
 
 
class SoftMaxModel(object):
    # 初始化参数
    def __init__(self,alpha=0.001,threhold=0.00005):
        self.alpha = alpha         # 学习率
        self.threhold = threhold   # 循环终止阈值
        self.num_classes = 3      # 分类数
     
    # 初始化权重，根据传传入参数x（样本）
    def setup(self,X):
        # 初始化权重矩阵，注意，由于是多分类，所以权重由向量变化为矩阵
        # 而且这里面初始化的是flat为1维的矩阵
        m, n = X.shape  # 400,15  
        s = math.sqrt(6) / math.sqrt(n+self.num_classes)
        theta = np.random.rand(n*(self.num_classes))*2*s-s  #[15,1]
        return theta
    
    # 计算样本X的softmax结果
    def softmax(self,x):
        # 先前传播softmax多分类
        # 注意输入的x是[batch数目n，类数目m]，输出是[batch数目n，类数目m]
        e = np.exp(x)
        temp = np.sum(e, axis=1,keepdims=True)
        return e/temp                           # 返回样本x在每个分类结果上的概率，矩阵形式
    
    # 计算cost 和 grad     
    def get_cost_grad(self,theta,X,y):
        m, n = X.shape
        theta_n = theta.reshape(n, self.num_classes)

        #计算Error，Cost，Grad
        y_dash = self.softmax(X.dot(theta_n))   # 批量向前传播结果
         
        Y = np.zeros((m,self.num_classes))                    # one-hot编码label矩阵
        for i in range(m):
            Y[i,y[i]]=1
 
        error = np.sum(Y * np.log(y_dash), axis=1)  # 通过矩阵计算，求每个样本的在每种分类上的cost
        cost = -np.sum(error, axis=0)               # 对每个样本的所有分类上的cost进行求和
        grad = X.T.dot(y_dash-Y)                    # 计算梯度 x*( 1(y_dash = k) - p(y=k) )
         
        grad_n = grad.ravel()                       # ravel() 将多维数组降为一维
        return cost,grad_n
         
         
  
    def train(self,X,y,max_iter=500, batch_size=64):
        m, n = X.shape  # 400,15
        theta = self.setup(X)
 
        #our intial prediction
        prev_cost = None
        loop_num = 0
        n_samples = y.shape[0]
        n_batches = n_samples // batch_size

        # Stochastic gradient descent with mini-batches
        while loop_num < max_iter:
            for b in range(n_batches):
                batch_begin = b*batch_size
                batch_end = batch_begin+batch_size
                X_batch = X[batch_begin:batch_end]
                Y_batch = y[batch_begin:batch_end]
                       
 
                #intial cost
                cost,grad = self.get_cost_grad(theta,X_batch,Y_batch)
                 
                theta = theta- self.alpha * grad/float(batch_size)
                 
            loop_num+=1
            if loop_num%10==0:
                print(cost,loop_num)
            if prev_cost:
                if prev_cost - cost <= self.threhold:
                    break
 
            prev_cost = cost
                        
             
        self.theta = theta
        print (theta,loop_num)
 
    def train_scipy(self,X,y):
        m,n = X.shape
        import scipy.optimize
        options = {'maxiter': 50, 'disp': True}
        J = lambda x: self.get_cost_grad(x, X, y)
        theta = self.setup(X)
 
        result = scipy.optimize.minimize(J, theta, method='L-BFGS-B', jac=True, options=options)
        self.theta = result.x
 
    def predict(self,X):
        m,n = X.shape
        theta_n = self.theta.reshape(n, self.num_classes)
        a = np.argmax(self.softmax(X.dot(theta_n)),axis=1)
        return a
 
 
    def grad_check(self,X,y):
        epsilon = 10**-4
        m, n = X.shape  
         
        sum_error=0
        N=300
         
        for i in range(N):
            theta = self.setup(X)
            j = np.random.randint(1,len(theta))
            theta1=theta.copy()
            theta2=theta.copy()
            theta1[j]+=epsilon
            theta2[j]-=epsilon
 
            cost1,grad1 = self.get_cost_grad(theta1,X,y)
            cost2,grad2 = self.get_cost_grad(theta2,X,y)
            cost3,grad3 = self.get_cost_grad(theta,X,y)
 
            sum_error += np.abs(grad3[j]-(cost1-cost2)/float(2*epsilon))
        print ("grad check error is %e\n"%(sum_error/float(N)))


def load_data(train_filename, test_filename):
    ########################## 从测试集中分离10%作为验证集
    train_dataset = h5py.File(train_filename, "r")
    x, y, z = train_dataset["train_set_x"].shape
    train = int(x * 0.9)
    test = x - train
    
    train_set_x_orig = np.array(
        np.concatenate((train_dataset["train_set_x"][:1500], train_dataset["train_set_x"][1700:3200],
                        train_dataset["train_set_x"][3400:4700]), axis=0))
    train_set_y_orig = np.array(
        np.concatenate((train_dataset["train_set_y"][:1500], train_dataset["train_set_y"][1700:3200],
                        train_dataset["train_set_y"][3400:4700]), axis=0))
    
    test_dataset = h5py.File(test_filename, "r")
    test_set_x_orig = np.array(
        np.concatenate((test_dataset["train_set_x"][1500:1700], test_dataset["train_set_x"][3200:3400],
                        test_dataset["train_set_x"][4700:]), axis=0))  # your test set features
    test_set_y_orig = np.array(
        np.concatenate((test_dataset["train_set_y"][1500:1700], test_dataset["train_set_y"][3200:3400],
                        test_dataset["train_set_y"][4700:]), axis=0))  # your test set labels
    ########################
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


if __name__=="__main__":
    
    train_filename = "MultiLabel_data_right.h5"
    print('Start read data')
    train_features, train_labels, test_features, test_labels = load_data(train_filename, train_filename)
    # 保持shape[0]不变，其他按适应性变化，即 保持样本个数不变
    train_X = train_features.reshape(train_features.shape[0],-1)  # The "-1" makes reshape flatten the remaining dimensions
    test_X = test_features.reshape(test_features.shape[0], -1)
    train_y = train_labels.reshape(train_labels.shape[0], -1)[0]  # The "-1" makes reshape flatten the remaining dimensions
    test_y = test_labels.reshape(test_labels.shape[0], -1)[0]
    
    # Standardize data to have feature values between 0 and 1.
    train_X = train_X / 255.
    test_X = test_X / 255.

    l_model = SoftMaxModel()
 
    l_model.grad_check(test_X[0:200,:],test_y[0:200])
 
    l_model.train(train_X, train_y, max_iter=1500, batch_size=512 )
    l_model.train_scipy(train_X, train_y)

    predict_train_y = l_model.predict(train_X)
    b = predict_train_y != train_y
 
    error_train = np.sum(b, axis=0)/float(b.size)   
 
    predict_test_y = l_model.predict(test_X)
    b = predict_test_y != test_y
 
    error_test = np.sum(b, axis=0)/float(b.size)
 
    print("Train Error rate = %.4f, \nTest Error rate = %.4f\n" % (error_train, error_test))