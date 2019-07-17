import cv2
import tensorflow as tf
import numpy as np

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

#数据随机选择函数
def get_batch(X,y,size):
    num = X.shape[0]
    index = np.random.choice(num,size,replace = False)
    return X[index],y[index]

#数据导入
data = np.load('data.npy')
label = np.load('labels.npy')

#分为训练集与测试集并归一化
num = int(data.shape[0] * 0.8)
X_tr = np.float32(data[0:num])/255.0
T_tr = label[0:num]
X_te = np.float32(data[num:-1])/255.0
T_te = label[num:-1]

#训练次数
num_batches = 200
#每次训练的数据量
batch_size = 128

#模型建立并优化
model = CNN()
optimizer = tf.train.AdamOptimizer(0.001)
print('----------------------------------')
print('please wait a moment.')
print('----------------------------------')
for i in range(num_batches):
    print('Number of runs is: %d/200' %i)
    X,y = get_batch(X_tr,T_tr,batch_size)
    with tf.GradientTape() as tape:
        y_pre = model(tf.convert_to_tensor(X))
        y = np.argmax(y,-1)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pre)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

#测试
y_pre = model.predict(X_te)
print('----------------------------------')
print('test accuracy: %f ' % (sum(np.argmax(y_pre,-1) == np.argmax(T_te,-1)) / X_te.shape[0]))
print('----------------------------------')
#保存模型
model.save_weights('./model/model')