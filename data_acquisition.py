import os
import cv2

#抓拍人的名字
name = 'wujiasheng'
#保存路径
path = 'faceImages/' + name
#生成文件夹
if not os.path.exists(path):
    os.makedirs(path)

cap=cv2.VideoCapture(0)
for i in range(600):
    #ret是返回下一帧，frame是读取该文件的内容
    ret ,frame = cap.read()
    #显示摄像头内容
    cv2.imshow('capture', frame)
    #保存摄像头内容
    cv2.imwrite(path + '/' + str(i)+'.jpg',frame)
    #等待20毫秒
    cv2.waitKey(20)
#关闭摄像头
cap.release()
cv2.destroyAllWindows()