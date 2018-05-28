import h5py

import keras
from keras.models import Sequential
from keras.models import load_model

def print_keras_wegiths(weight_file_path):
    f = h5py.File(weight_file_path,'r')  # 读取weights h5文件返回File类
    try:
        if len(f.attrs.items()):
            print("{} contains: ".format(weight_file_path))
            print("Root attributes:")
        for key, value in f.attrs.items():
            print("  {}: {}".format(key, value))  # 输出储存在File类中的attrs信息，是各层的结构、配置以及模型的优化方式

        for layer, g in f.items():  # 读取各层的名称以及包含层信息的Group类
            print("  {}".format(layer))
            print("    Attributes:")
            for key, value in g.attrs.items(): # 输出储存在Group类中的attrs信息，一般是各层的weights和bias及他们的名称
                print("      {}: {}".format(key, value))

            print("    Dataset:")
            for name, d in g.items(): # 读取各层储存具体信息的Dataset类
                print("      {}: {}".format(name, d.items())) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                # print("      {}: {}".format(name, d.value.shape)) # 输出储存在Dataset中的层名称和权重，也可以打印dataset的attrs，但是keras中是空的
                # print("      {}: {}".format(name. d.value))
    finally:
        f.close()



def keras_model(weight_file_path):
    f = h5py.File(weight_file_path,'r')  # 读取weights h5文件返回File类
    
    nn_model = load_model(weight_file_path)
    print(nn_model.get_config())
    print(nn_model.summary())
    for layer in nn_model.layers:
        print(layer)
        # print(nn_model.get_layer("conv2d_1"))
    
    model_weights = nn_model.get_weights()
    return model_weights


def writefile(filepath, content):
    try:
        file = open(filepath,'a')
    except IOError:
        print("*** file open error!")
    else:
        file.write(content)
        file.close()
        
def writepara(filename, para, model_weights):
    i = 0
    for weight in model_weights:
        print(weight.shape)
        if para[i] == "conv2d_1":
            conv1 = weight[:, :, :, 0]
            conv2 = weight[:, :, :, 1]
            print(conv1)
            print(conv2)
            # writefile(filename+"conv2d_1.txt", para[i] + '\n')
            writefile(filename+"conv2d_1.txt", str(conv1) + '\n')
            writefile(filename+"conv2d_1.txt", str(conv2) + '\n')
        elif para[i] == "conv2d_1_bias":
            conv_bias = weight[:]
            print(conv_bias)
            # writefile(filename+"conv2d_1_bias.txt", para[i] + '\n')
            writefile(filename+"conv2d_1_bias.txt", str(conv_bias) + '\n')
        
        if para[i] == "conv2d_2":
            conv1 = weight[:, :, :, 0]
            conv2 = weight[:, :, :, 1]
            print(conv1)
            print(conv2)
            # writefile(filename+"conv2d_2.txt", para[i] + '\n')
            writefile(filename+"conv2d_2.txt", str(conv1) + '\n')
            writefile(filename+"conv2d_2.txt", str(conv2) + '\n')
        elif para[i] == "conv2d_2_bias":
            conv_bias = weight[:]
            print(conv_bias)
            # writefile(filename+"conv2d_2_bias.txt", para[i] + '\n')
            writefile(filename+"conv2d_2_bias.txt", str(conv_bias) + '\n')
        
        if para[i] == "dense_1":
            # writefile(filename+"dense_1.txt", para[i] + '\n')
            for k in range(weight.shape[0]):
                print(weight[k])
                writefile(filename+"dense_1.txt", str(weight[k]) + '\n')
        elif para[i] == "dense_1_bias":
            # writefile(filename+"dense_1_bias.txt", para[i] + '\n')
            print(weight[:])
            writefile(filename+"dense_1_bias.txt", str(weight[:]) + '\n')
        
        if para[i] == "dense_2":
            # writefile(filename+"dense_2", para[i] + '\n')
            for k in range(weight.shape[0]):
                print(weight[k])
                writefile(filename+"dense_2.txt", str(weight[k]) + '\n')
        elif para[i] == "dense_2_bias":
            # writefile(filename+"dense_2_bias.txt", para[i] + '\n')
            print(weight[:])
            writefile(filename+"dense_2_bias.txt", str(weight[:]) + '\n')

        i += 1
        print("******************")
    
if __name__ == '__main__':
    
    # print_keras_wegiths("./my_model_acc0.995.h5")
    model_weights = keras_model("./my_model_acc0.995.h5")
    print(len(model_weights))
    para = ["conv2d_1","conv2d_1_bias", "conv2d_2", "conv2d_2_bias",
            "dense_1", "dense_1_bias", "dense_2", "dense_2_bias"]
    
    writepara(filename="./paras/", para=para, model_weights=model_weights)
    
    