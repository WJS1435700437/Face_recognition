import numpy as np
import os
import cv2

def img2array(path):
    '''
    Define img2array function
    
    Parameters
    -----------------------------------------------
    path : list
           the path of image,the length of the list is the number of label
           size: 10
    Return 
    ------------------------------------------------
    data : np.array
           the data of image
           size: [image_num,120,90,1]
    label: np.array(one hot)
           the label of image,
           size: [image_num,10]          
    '''
    data = []
    label = []
    label_num = len(path)
    for i in range(label_num):
        imgnames = os.listdir(path[i])
        print(i,path[i])
        for imgname in imgnames:
            img = cv2.resize(cv2.imread(path[i] + '/' + imgname,0),(120,90)).reshape([120,90,1])    
            data.append(img)
            label.append(np.eye(label_num)[i])
    num = len(data)
    index = np.random.choice(num,num,replace = False)
    data = np.uint8(np.array(data)[index])
    label = np.uint8(np.array(label)[index])
    return data,label

path = 'faceImageGray/'
name = ['huajinqing', 'liangchunfu', 'lijunyu', 'linjuncheng','linweixin',
        'wujiasheng', 'xuhaolin', 'zenglingqi', 'zhouyuanxiang', 'zhushichao']
filepath = [path + n for n in name]
data,labels = img2array(filepath)
np.save('data.npy',data)
np.save('labels.npy',labels)