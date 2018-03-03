import cv2
import numpy as np
import os

print(cv2.__version__)

def video2frame(videopath,videoname):
    video = videopath + videoname + ".mp4"          # 找到具体的视频
    vidcap = cv2.VideoCapture(video)                # 使用VideoCapture() 函数进行视频帧捕获
    print(vidcap.isOpened())
    success, image = vidcap.read()                  # 读取捕获的每一帧，如果有帧率，返回 True 和 该帧图像
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        print('Read a new frame: %d' % count, success)
        framename = videoname + "_" + "%d.jpg" %(count)     # 自定以帧的保存格式
        savepath = videopath + 'frames/'                    # 保存路径设置
        cv2.imwrite(savepath+framename, image)              # 保存路径和帧名字组合，可以将截取的帧保存到指定位置
        count += 1
        
if __name__ == '__main__':
    dirlist = ['left','midle','right']          # 三类视频文件夹名称
    for dir in dirlist:                         # 统计每一类文件夹中视频个数
        dirsum = len(os.listdir("./" + dir))    # os.listdir(path)语句
        print(dir + "\tvideo number is:" +str(dirsum))
        for i in range(dirsum):                        # 依次处理文件夹下面的视频，做提取帧
            videopath = "./" + dir + "/"               # 生成视频存储路径
            videoname = dir + str(i)                   # 生成视频名字，为后续处理具体视频做铺垫
            video2frame(videopath = videopath, videoname = videoname)
    # video2frame("./right/right1.mp4")



'''
img = cv2.imread("./train_set_yesAndno/0.jpg",0)

# cv2.imshow("dd",img)

cap = cv2.VideoCapture("./right/right1.mp4")
r,img = cap.read()
print(r)
'''