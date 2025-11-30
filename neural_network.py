#感知机和神经网络的差异性
#实际上：感知机可以理解为简单且特殊的神经网络，感知机即使用阶跃函数的单层神经网络，如果感知机使用其他激活函数就进入了神经网络
#由于阶跃函数的性质，感知机的神经元之间流动的都是0/1，而神经网络的神经元之间流动的是连续的实数（即二者在平滑性方面有差异，而神经网络的平滑性在在学习中非常重要）
#单层感知机只能解决简单的线性问题，而多层神经网络能够解决更有深度更加抽象的非线性问题，随着深度的逐渐提升，多层神经网络能够提取数据中更加抽象的特征
#重点理解------------#对于为什么单层感知机中的激活函数阶跃函数是非线性的，而解决不了非线性问题，因为能否解决什么样的问题取决于决策边界
#决策边界是线性的，就只能解决线性问题，决策边界非线性，就能解决非线性问题
#这里需要对激活函数和决策边界进行进一步的阐述：决策边界是神经元中的权重和（W*X+b=0，W，X为列向量）的x所对应点的集合，也就是超平面，即永远是线性的
#也就是说每个神经元的决策边界都是线性的，只有当多个神经元进行组合时才有可能生出非线性的决策边界
#对于激活函数的理解，其实他只是对权重和进行阈值化，
#对激活函数和决策边界进行联系理解：决策边界类似一把线性刀，先把空间进行线性切割为两部分空间（这一步与激活函数无关），而激活函数则是把两部分空间的值进行阈值化并且决定取哪部分空间
#接着输出层的会把被选取的空间进行整合从而实现决策边界的非线性
#以上其实也与激活函数必须是非线性的相互对应，因为如果激活函数是线性的，那么无论多少层神经网络或者说神经元叠加都是简单的线性叠加，最后还是会生成线性决策边界






#阶跃函数
#传入实数的实现
import numpy as np
import matplotlib.pylab as plt

def step_function_r(x):
    if x>0:
        return 1
    else:
        return 0
#传入数组的实现
def step_function_ar(x):
    y=x>0#等价return np.array(x>0,dtype=np.int)
    return y.astype(np.int_)#利用。astype方法转换数组的数据类型

x=np.array([1,2,3])
print(step_function_ar(x))

x=np.arange(-5.0,5.0,0.1)#np.arange(x,y,z)只能生成一维数组，x,y确定范围，z确定步长，虽然只能生成一维数组，但是后续可以通过reshape方法指定转化为几维
y=step_function_ar(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)#指定y轴的范围
plt.show()

#sigmoid函数
def sigmoid(x):
    return 1/(np.exp(-x)+1)

x=np.arange(-5,5,0.1)
y=sigmoid(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()

# ReLUh函数
def ReLU(x):
    return np.maximum(0,x)

x=np.arange(-5,5,0.1)
y=ReLU(x)
plt.plot(x,y)
plt.show()

#恒等函数
def identity_function(x):
    return x
#三层神经网络的实现
def init_network():
    network={}
    network['W1']=np.array([[0.1,0.3,0.5],
                            [0.2,0.4,0.6]])
    network['B1']=np.array([[0.1,0.2,0.3]])
    network['W2']=np.array([[0.1,0.4],
                            [0.2,0.5],
                            [0.3,0.6]])
    network['B2']=np.array([0.1,0.2])
    network['W3']=np.array([[0.1,0.3],
                            [0.2,0.4]])
    network['B3']=np.array([0.1,0.2])
    return network

def forword(x,network):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    B1,B2,B3=network['B1'],network['B2'],network['B3']

    A1=np.dot(x,W1)+B1
    h1=sigmoid(A1)
    A2=np.dot(h1,W2)+B2
    h2=sigmoid(A2)
    h3=np.dot(h2,W3)+B3
    y=identity_function(h3)
    return y

x=np.array([1.0,0.5])
network= init_network()
y=forword(x,network)

print(y)

#softmax函数:特征：输出层的总和永远为一，因此可以用来表示概率问题，数值最大的一项输出（最有可能正确），又因为softmax函数是递增函数，所以
#没有经过softmax激活的原值也是最大的那一项，所以有时是可以省略softmax函数的

def softmax(x):
    #return np.exp(x)/np.sum(np.exp(x))
    #return np.exp(x-np.max(x))/np.sum(np.exp(x-np.max(x)))
    max_exp=np.max(x)
    exp=np.exp(x-max_exp)
    sum_exp=np.sum(exp)
    return exp/sum_exp

x=np.array([1010,1000,990])
print(softmax(x))

#输出层和输入层的神经元数量是由问题来决定的，首先对于分类问题，一般有几类输出层就有几个神经元
#对于输入层，则需要看输入数据转化为数组形式后存在几组
#例如：识别mnist图像，训练图库有60000张，而每一张的像素为28*28，因此，当一张图片的数据转化为数组形式时
#可以用28*28个神经元来代替，因此如果要训练一张照片则输入层为784个神经元，即一个包含784个元素的一维数组
#如果要训练一整个60000的图库，则可以用二维数组表示，一维有60000个元素，代表60000张图片，二维有784个元素，代表一张照片的数据



'''#mnist图像复现
import sys,os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image


(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

'''

#显示mnist图库的第一个图像
import os,sys
sys.path.insert(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image

(x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False)
img=x_train[0]
img=img.reshape(28,28)
def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))#Image.fromarray是把numpy中的数组转化为PIL(pillow)中的图像对象，而且它必须接受特定的数据类型如uint8
    pil_img.show()
img_show(img)

#神经网络的推理处理
def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,flatten=True,one_hot_label=False)#flatten,进行高维数组展平处理，normalize即正规化，一种预处理，即对输入数据进行处理
    return x_test,t_test

def init_network():
    with open('sample_weight.pkl','rb') as f:#此处是加载pickle文件
        #对pickle文件进行简单介绍，一共存在两个步骤：保存，加载pickle文件，其中存在序列化的过程，有两种pickle和joblib序列化（此处用的是pickle序列化）
        #保存：import pickle/joblib
        #      data=[一系列数据]
        #     with open ('sample_weight.pkl','wb') as f:
        #       pickle/joblib.dump(data,f)
        #加载：import pickle/joblib
        #   with open ('sample_weight',"rb") as f:
        #   data=pickle.load(f)
        #其中“rb”“wb”分别是按照二进制读取和写入文件
        network=pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    B1,B2,B3=network['B1'],network['B2'],network['B3']

    A1=np.dot(x,W1)+B1
    h1=sigmoid(A1)
    A2=np.dot(h1,W2)+B2
    h2=sigmoid(A2)
    h3=np.dot(h2,W3)+B3
    y=softmax(h3)
    return y

x,t=get_data
network=init_network

for i in range(len(x)):
    y=predict(network,x[i])
    p=np.argmax(y)
    if p==t[i]:
        accuracy_cnt+=1

print("Accuracy:"+str(float(accuracy_cnt)/len(x)))

#批量处理
for i in range(0,len(x),100):
    y=predict(network,x[i:(i+100)])
    p=np.argmax(y,axis=1)#argmax,求数组中最大值的下标，axis，说明移动的方向，1即一维，代表行向量，0即0维，代表列向量
    accuracy_cnt+=np.sum(p==t[i:(i+100)])#知识点：np.sum可以求和布尔类型的数组，flase为0，True为1，数组和数组进行数值运算符的结果是布尔类型的数组

