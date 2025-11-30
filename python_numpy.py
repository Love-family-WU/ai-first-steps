'''收获和思考：
1.如何在终端处打开python文件
2.广播的理解：两个形状不同的数组，其中一个数组会自动广播成适用于两个数组相乘的形式
广播的方式是在原有的基础上复制向需要广播的方向填充，注意：广播不仅适用于乘法，它适用于四则运算
注意:广播的局限性：A.shape=(m,n),B,A*B能够广播的条件是B必须是单行或者单列（一维数组或者二维数组的行向量x或列向量y）且x<m or y<n
3.关于函数与方法区别的思考：方法即是存在在类中的一个函数
4.关于类的定义的思考：其中存在————init————（self，形参）函数，该函数只有在类中需要传递实参的时候定义，
什么时候类需要传递实参呢，当需要一个贯穿类中的所有函数都需要用到该变量的时候在用，其中self.变量称为实例变量
需要用--init--中的形参初始化self.变量
5.考虑列表和数组的区别：数组存在相应元素的加减乘除和广播，而列表只有加法而且是直接拼接
6.对于一维数组想要获取某些特征元素时需注意：传入元素下标时，要传入数组形式而不是列表
如：a[np.array([0,2,4])],a[a>4]
其中注意：a>4，同样生成的是数组，不过是布尔型的数组，a[a>4]会返回为true的元素'''










#numpy的运用
#函数方法补充：np.ndim()表示数组维度，np.astype()将布尔型数组转化为整型
import numpy as np
import matplotlib.pyplot as plt
#广播实验
a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9]])
z=np.array([[1,2,3]
            ])

print(a*z)

a=a.flatten()
print(a)
print(a>4)
print(a[np.array([0,2,4])])

b=[1,2,3]
c=[1,2,3]
print(b+c)

print(type(a>4))

x=np.arange(0,6,0.00001)#生成一个以0开始步长为0.00001以5.99999结尾的数组
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,label="sin")
plt.plot(x,y2,linestyle='--',label='cos')
plt.xlabel('x')
plt.ylabel('y')
plt.title('sin&cos')
plt.legend()
plt.show()

#matplotlib的运用
import matplotlib.pyplot as plt
from matplotlib.image import imread

rng=imread('d:\\xwechat_files\\wxid_fru6qxx8z4aa22_3bfe\\temp\RWTemp\\2025-11\\a903c210a026cab0926f6fa0575f7b5d.jpg')

plt.imshow(rng)
plt.show()

#关于多维数组的实验
m=np.array([1,2,3,4])
print(m.shape)
print(np.ndim(m))
m=np.array([[1,2,3,4]])
print(m.shape)
print(np.ndim(m))
#结果可见，对于一位数组的形状结果实际意义即存在几个元素，但是结果的形式
#必须和多维数组的形式保持一致，所以为（4，），但是当把它塑造为二位数组时就成了我们常见的行向量的形式及一行四列(1,4)
#如果用机器内部的理解形式解释：
#对于数组np.array([[1,2,3,4]])的shape为（1，4）即第一维数组有一个元素及[1,2,3,4]
#第二维数组[1,2,3,4]有四个元素及1，2，3，4，显而易见有几个[]就存在几个数组，从最外层数组开始理解为第一维数组
