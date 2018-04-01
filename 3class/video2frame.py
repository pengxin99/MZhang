import cv2
import numpy as np
import os
from PIL import Image

print(cv2.__version__)


def video2frame(videopath,videoname):
    """
    Parameters
        ----------
        videopath : 视频存放路径
        videoname : 视频名字，用来找到具体的视频
    """
    video = videopath + videoname + ".mp4"          # 找到具体的视频
    print("start ***********" + video)
    vidcap = cv2.VideoCapture(video)                # 使用VideoCapture() 函数进行视频帧捕获
    print(vidcap.isOpened())
    success, image = vidcap.read()                  # 读取捕获的每一帧，如果有帧率，返回 True 和 该帧图像

    count = 0
    imgsave_sum = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: %d' % imgsave_sum, success)
        if count % 2 == 0 :                                           # 每两帧保存一帧图像
            imgsave_sum += 1
            framename = videoname + "_" + "%d.jpg" %(imgsave_sum)     # 自定以帧的保存格式
            savepath = videopath + 'frames/'                    # 保存路径设置
            
            # 此处先将image从 array 转到 img，进行resize操作； 然后再从img 转到 array ，进行imwrite操作
            # 遇到错误选择跳过 继续后续执行
            try:
                img_resize = Image.fromarray(image).resize((100, 100),
                                                           Image.ANTIALIAS)  # resize image with high-quality
            except AttributeError:
                print(" 'NoneType' object has no attribute '__array_interface__'")
                continue
            cv2.imwrite(savepath+framename, np.array(img_resize))              # 保存路径和帧名字组合，可以将截取的帧保存到指定位置
        count += 1

    print("END ################# ***********" + video)
        
if __name__ == '__main__':
    dirlist = ['left','midle','right']          # 三类视频文件夹名称
    for dir in dirlist:                         # 统计每一类文件夹中视频个数
        dirsum = len(os.listdir("./" + dir))    # os.listdir(path)语句
        print(dir + "\tvideo number is:" +str(dirsum))
        for i in range(dirsum):                             # 依次处理文件夹下面的视频，做提取帧
            videopath = "./" + dir + "/"                    # 生成视频存储路径
            videoname = dir + str(i + 1)                    # 生成视频名字，为后续处理具体视频做铺垫
            video2frame(videopath = videopath, videoname = videoname)


