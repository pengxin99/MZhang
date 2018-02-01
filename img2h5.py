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


# 将图片信息转为h5文件
# filename：最终输出打文件名
# picfile：图片存放地址
# num: 文件夹中共有多少张照片
def pic_2_h5(filename,picfile_path,label = True,num = 200):

    # 创建文件
    f = h5py.File(filename + ".h5", 'w')

    sets_x = []
    sets_y = []
    # 读取200 张图片，存入h5文件
    for i in range(num):
        print(str(i) + '\n')
        pic_name = picfile_path + "/" + str(i) + ".jpg"

        image = np.array(ndimage.imread(pic_name, flatten=False, mode= 'L'))
        # num_px = image.shape[0]
        # num_py = image.shape[1]
        # my_image = scipy.misc.imresize(image, size=(num_px, num_py)).reshape((num_px * num_py * 3, 1))
        sets_x.append(image)
        if i < num / 2:
            sets_y.append( 1  )
            print("#")

        else:
            sets_y.append( 0  )
            print("*")

    data = np.array(sets_x)
    labels = np.array(sets_y)
    print(data.shape)

    dset1 = f.create_dataset("train_set_x", data = data)
    dset2 = f.create_dataset("train_set_y", data = labels)
    '''
     for key in f.keys():
        print(f[key].name)
        print(f[key].value)
        print(type(f[key].value))
    '''


if __name__ == '__main__':

    # resizePic("testPic","testPic_resize",num=200, width=100, height=100)
    # pic_2_h5("test_mdvsnmd", "testPic_resize")

    # resizePic("test_set_pic", "test_set_pic_resize", num=11, width=100, height=100)
    pic_2_h5("train_set_all_MODE_L", "train_set_yesAndno",num=400)

    print("\n\nEND!")