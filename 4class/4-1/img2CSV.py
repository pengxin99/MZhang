
import numpy as np

from PIL import Image
from scipy import ndimage
import os.path

import csv



# 将图片信息转为CSV文件
# filename：最终输出的文件名
# picfile：图片存放地址

def pic_2_csv(filename, picfile_path):
    
    # 打开csv文件
    out = open(filename, "w")
    
    # 图片的标签
    frame_labels = {'left': 0, 'midle': 1, 'right': 2, 'point': 3}
    
    # 读取4种类别的图片，依次读取，将图片数据和标签共同存入image
    dirlist = ['left', 'midle', 'right', 'point']               # 三类视频文件夹名称
    for dir in dirlist:                                         # 统计每一类文件夹中视频个数
        frame_path = picfile_path + dir + "/frames/"
        frames = []
        
        # 找到当前类别下的所有帧图片，将名字存入frames列表中
        for root, dirs, files in os.walk(frame_path):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':         # 其中os.path.splitext()函数将路径拆分为文件名+扩展名
                    # frames.append(os.path.join(root, file))
                    frames.append(file)
        
        for frame in frames:
            image = np.array(ndimage.imread(frame_path + frame, flatten=False, mode='L'))
            image = image.reshape(1, 64*64)[0].tolist()         # 此处reshape，因为后续使用keras、cnn时，需要有图片的通道信息；tolist()是因为要在后面追加label
            # print(type(image))
            # print(image)
            image.append(frame_labels[dir])
            
            # 打印当前执行的是哪个类别的图片
            if frame_labels[dir] == 0:
                print("left! \n")
            elif frame_labels[dir] == 1:
                print("midle! \n")
            elif frame_labels[dir] == 2:
                print("right! \n")
            else:
                print("point! \n")
        
            # 设定写入模式
            csv_write = csv.writer(out)
            # 写入具体内容
            csv_write.writerow(image)
        
    out.close()

if __name__ == '__main__':
    
    filename = "test.csv"
    video_path = "./"
    pic_2_csv(filename, video_path)
 
    
    print("\n\nEND!")