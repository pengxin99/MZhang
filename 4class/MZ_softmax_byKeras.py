import numpy as np

# np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import matplotlib.pyplot as plt


from keras.models import load_model

from util import *
### keras visualization
from keras.utils import plot_model


def makeModel(kernel_size1, kernel_size2):
    # 构建模型，序贯模型
    model = Sequential()
    
    model.add(Convolution2D(nb_filters, (kernel_size1[0], kernel_size1[1]),
                            padding='same',
                            input_shape=input_shape))  # 卷积层1
    model.add(Activation('tanh'))  # 激活层
    model.add(Convolution2D(nb_filters, (kernel_size2[0], kernel_size2[1])))  # 卷积层2
    model.add(Activation('tanh'))  # 激活层
    model.add(MaxPooling2D(pool_size=pool_size))  # 池化层
    # model.add(Dropout(0.25))                          # 神经元随机失活
    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(128))  # 全连接层1
    model.add(Activation('tanh'))  # 激活层
    # model.add(Dropout(0.5))                           # 随机失活
    model.add(Dense(nb_classes))  # 全连接层2
    model.add(Activation('softmax'))  # Softmax评分
    
    return model

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


if __name__ == '__main__':
    
    # 全局变量
    batch_size = 128
    nb_classes = 4
    epochs = 20
    # input image dimensions
    img_rows, img_cols = 64, 64
    # number of convolutional filters to use
    nb_filters = 2
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size1 = (5, 5)
    kernel_size2 = (21, 21)
    
    # 加载图片数据
    # the data, shuffled and split between train and test sets
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_filename = "train.csv"
    test_filename = "./4-1/test.csv"
    (X_train, y_train, X_test, y_test) = load_data_csv(train_filename, test_filename)
    print('Y_train shape:', y_train.shape)
    print('Y_test shape:', y_test.shape)
    
    # 根据不同的backend定下不同的数据格式（theano和tensorflow输入数据的格式不同）
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', y_train.shape)
    print('Y_test shape:', y_test.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    
    # 转换为one_hot类型
    Y_train = np_utils.to_categorical(y_train[0], nb_classes)           # 此处使用y_train[0]，是为了避免，读入数据集的维度不同。这里只需要一维数据
    Y_test = np_utils.to_categorical(y_test[0], nb_classes)
    print('Y_train shape:', Y_train.shape)
    print('Y_test shape:', Y_test.shape)
    '''
    
    # 构建模型，序贯模型
    model = Sequential()
    model = makeModel(kernel_size1, kernel_size2)
    
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # 创建一个实例history
    history = LossHistory()
    # 训练模型（注意这个地方要加入callbacks）
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[history])
   
    # 评估模型
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    # 保存模型
    # model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
    
    '''
    
    # 从加载的模型进行预测
    
    model = load_model('my_model_acc0.995.h5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    prediction = model.predict(X_test)
    
    # 预测的输出结果进行标准化，因为预测输出结果是每个分类的概率，这里转换为[1,0,0,0] one-hot 形式
    for i in range(prediction.shape[0]):
        index = 0
        temp = 0
        for j in range(prediction[i].shape[0]):
            if( prediction[i][j] > temp):
                temp = prediction[i][j]
                index = j
        for j in range(prediction[i].shape[0]):
            if j != index:
                prediction[i][j] = 0
            else:
                prediction[i][j] = 1
        


    
    print_mislabeled_images(X_test, Y_test, prediction)       # 显示错误分类的图像
    
    # 可视化模型（模型结构）
    plot_model(model, to_file='cnn+softmax_model.png', show_shapes = True)

    

    # 绘制acc-loss曲线
    # history.loss_plot('epoch')
