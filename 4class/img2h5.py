import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import os.path

# f=h5py.File("myh5py.hdf5","w")
# #分别创建dset1,dset2,dset3这三个数据集
# a=np.arange(20)
# d1=f.create_dataset("dset1",data=a)
#
# d2=f.create_dataset("dset2",(3,4),'i')
# d2[...]=np.arange(12).reshape((3,4))
#
# f["dset3"]=[1,2,3]
#
# for key in f.keys():
#     print(f[key].name)
#     print(f[key].value)
#     print(type(f[key].value))

# This is the function introduction of imread():
"""
    Read an image from a file as an array.

    Parameters
    ----------
    name : str or file object
        The file name or file object to be read.
    flatten : bool, optional
        If True, flattens the color layers into a single gray-scale layer.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more
        details.

    Returns
    -------
    imread : ndarray
        The array obtained by reading the image.

    Notes
    -----
    `imread` uses the Python Imaging Library (PIL) to read an image.
    The following notes are from the PIL documentation.

    `mode` can be one of the following strings:

    * 'L' (8-bit pixels, black and white)
    * 'P' (8-bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8-bit pixels, true color)
    * 'RGBA' (4x8-bit pixels, true color with transparency mask)
    * 'CMYK' (4x8-bit pixels, color separation)
    * 'YCbCr' (3x8-bit pixels, color video format)
    * 'I' (32-bit signed integer pixels)
    * 'F' (32-bit floating point pixels)

    PIL also provides limited support for a few special modes, including
    'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
    (true color with premultiplied alpha).

    When translating a color image to black and white (mode 'L', 'I' or
    'F'), the library uses the ITU-R 601-2 luma transform::

        L = R * 299/1000 + G * 587/1000 + B * 114/1000

    When `flatten` is True, the image is converted using mode 'F'.
    When `mode` is not None and `flatten` is True, the image is first
    converted according to `mode`, and the result is then flattened using
    mode 'F'.

    """


'''
def resizePic(inpath,outpath,num = 100, width = 100, height = 100):
    for i in range(num):
        infile = inpath + '/' + str(i)+'.jpg'  #输入图片所在路径
        outfile = outpath + '/' + str(i)+'.jpg' #输出图片所在路径
        # Image.open(infile).convert('RGB').save(outfile) # convert the image to RGB mode  very important 将图片转化为RGB模式，虽然我不知道为什么，但是没有这一步会报错
        im = Image.open(infile)
        (x, y) = im.size             # read image size
        x_s = width                 # define standard width
        y_s = height                 # calc height based on standard width
        out = im.resize((x_s, y_s), Image.ANTIALIAS)    # resize image with high-quality
        out.save(outfile)
        print(str(i+1)+'original size: ', x, y)
        print(str(i+1)+'adjust size: ', x_s, y_s)
'''

# 将图片信息转为h5文件
# filename：最终输出的文件名
# picfile：图片存放地址
# num: 文件夹中共有多少张照片
def pic_2_h5(filename, picfile_path, label = True):
    
    sets_x = []
    sets_y = []
    frame_labels = {'left':0, 'midle':1, 'right':2, 'point':3}
    
    # 读取3种类别的图片，依次读取1600张，存入sets_x、sets_y矩阵
    dirlist = ['left', 'midle', 'right', 'point']                        # 三类视频文件夹名称
    for dir in dirlist:                                         # 统计每一类文件夹中视频个数
        frame_path = picfile_path + dir + "/frames/"
        frames = []
        
        # 找到当前类别下的所有帧图片，将名字存入frames列表中
        for root, dirs, files in os.walk(frame_path):
            for file in files:
                if os.path.splitext(file)[1] == '.jpg':             # 其中os.path.splitext()函数将路径拆分为文件名+扩展名
                    # frames.append(os.path.join(root, file))
                    frames.append(file)
        
        for frame in frames:
            image = np.array(ndimage.imread(frame_path+frame, flatten=False, mode='L'))
            image = image.reshape(1,100,100)            # 此处reshape，因为后续使用keras、cnn时，需要有图片的通道信息
            print(image.shape)
            sets_x.append(image)
            sets_y.append(frame_labels[dir])
            if frame_labels[dir] == 0 :
                print("left! \n")
            elif frame_labels[dir] == 1 :
                print("midle! \n")
            elif frame_labels[dir] == 2 :
                print("right! \n")
            else:
                print("point! \n")

        data = np.array(sets_x)
        labels = np.array(sets_y)
        print(data.shape)
        
        # 创建h5文件
        f = h5py.File(filename+ "_" + dir + ".h5", 'w')
        dset1 = f.create_dataset("train_set_x", data = data)
        dset2 = f.create_dataset("train_set_y", data = labels)

if __name__ == '__main__':
    
    filename = "MultiLabel_data"
    video_path = "./"
    pic_2_h5(filename, video_path)

    print("\n\nEND!")