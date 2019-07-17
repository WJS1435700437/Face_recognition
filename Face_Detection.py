from mtcnn.mtcnn import MTCNN
import cv2
import os

path = 'faceImages'
new_path = 'faceImageGray'
files = os.listdir(path)
for file in files:
    Images = os.listdir(path + '/' + file)
    if not os.path.exists(new_path + '/' + file):
        os.makedirs(new_path + '/' + file)
    for imgname in Images:
        img = cv2.imread(path + '/' + file + '/' + imgname)
        detector = MTCNN()
        result = detector.detect_faces(img)
        if result:
            x1 = result[0]['box'][1]
            x2 = result[0]['box'][3] + x1
            y1 = result[0]['box'][0]
            y2 = result[0]['box'][2] + y1
            mt_img = img[x1:x2,y1:y2]
            imggray = cv2.cvtColor(mt_img,cv2.COLOR_BGR2GRAY)
            k = cv2.imwrite(new_path + '/' + file + '/' + imgname ,imggray)