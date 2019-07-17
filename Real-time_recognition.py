import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import cv2

tf.enable_eager_execution()

#卷积神经网络
class CNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = conv2d(16,[3,3],activation = tf.nn.relu,name = 'conv1')
        self.pool1 = pool2d([2,2],name = 'pool1')
        self.conv2 = conv2d(32,[3,3],activation = tf.nn.relu,name = 'conv2')
        self.pool2 = pool2d([2,2],name = 'pool2')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128,activation = tf.nn.relu,name = 'dense1')
        self.dense2 = tf.keras.layers.Dense(10,name = 'dense2')
    
    def call(self,inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x
    
    def predict(self,inputs):
        pre = self(inputs)
        return pre.numpy()
    
    def restore(self,path):
        self.load_weights(path)
        
#卷积层重新包装        
def conv2d(filters,size,strides = [1,1],padding = 'same',activation = None,name = 'conv'):
    return tf.keras.layers.Conv2D(filters = filters,
                                 kernel_size = size,
                                 strides = strides,
                                 padding = padding,
                                 activation = activation,
                                 name = name)

#池化层重新包装
def pool2d(size,strides = [2,2],padding = 'same',name = 'pool'):
    return tf.keras.layers.MaxPool2D(pool_size = size,strides = strides,
                                     padding = padding,name = name)

#softmax函数
def softmax(x):
    x = np.exp(x - np.max(x))
    return x / np.sum(x)

#建立卷积神经网络模型
model = CNN()
#导入权重参数
model.restore('./model/model')
#建立mtcnn模型
detector = MTCNN()

name = ['huajinqing', 'liangchunfu', 'lijunyu', 'linjuncheng','linweixin',
        'wujiasheng', 'xuhaolin', 'zenglingqi', 'zhouyuanxiang', 'zhushichao']

#实时抓拍
cap=cv2.VideoCapture(0)
while True:
    ret ,frame = cap.read()
    #切换通道
    img = frame[:,:, (2, 1, 0)]
    #人脸检测
    result = detector.detect_faces(frame)
    if result:
        #框出人脸图
        x1 = result[0]['box'][1]
        x2 = result[0]['box'][3] + x1
        y1 = result[0]['box'][0]
        y2 = result[0]['box'][2] + y1
        if x1 < 0 or y1 < 0:
            continue
        mt_img = img[x1:x2,y1:y2]
        #灰度化
        imggray = cv2.cvtColor(mt_img,cv2.COLOR_BGR2GRAY)
        #预测
        img = np.float32(cv2.resize(imggray,(120,90)).reshape([1,120,90,1]))/255.0
        pre = model.predict(img)
        pre = softmax(pre.reshape([-1]))
        #对于识别概率低于0.8识别为unknown
        if np.max(pre) < 0.8:
            cv2.rectangle(frame,(y1,x1), (y2,x2), (0,255,0), 2)
            cv2.putText(frame, 'unknow', (y1, x1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        #其他按标签输出
        else:
            pre = np.argmax(pre)
            cv2.rectangle(frame,(y1,x1), (y2,x2), (0,255,0), 2)
            cv2.putText(frame, name[pre], (y1, x1), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
    #显示图片
    cv2.imshow('capture', frame)
    #停顿10毫秒与按q退出抓拍
    if cv2.waitKey(10) &  0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()