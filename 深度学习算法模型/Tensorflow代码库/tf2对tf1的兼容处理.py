#引入TensorFlow库
import tensorflow as tf
#import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()

#引入数值计算库
import numpy as np

#使用 NumPy 生成假数据集x,代表房间的平米数，这里的取值范围是0-1的浮点数，
#原因请看正文中的说明，属于是“规范化”之后的数据
# 生成的数据共100个，式样是100行，每行1个数据
x = np.float32(np.random.rand(100,1))
#我们假设每平米0.5万元，基础费用0.7万，这个数值也是规范化之后的，仅供示例
#最终运行的结果，应当求出来0.5/0.7这两个值代表计算成功
#计算最终房价y，x和y一同当做我们的样本数据
# np.dot的意思就是向量x * 0.5
y = np.dot(x,0.5) + 0.7
#---------------------------------数据集准备完成
#以下使用TensorFlow构建数学模型，在这个过程中，
#直到调用.run之前，实际上都是构造模型，而没有真正的运行。
#这跟上面的numpy库每一次都是真正执行是截然不同的区别
# 请参考正文，我们假定房价的公式为：y=a*x+b

#tf.Variable是在TensorFlow中定义一个变量的意思
#我们这里简单起见，人为给a/b两个初始值，都是0.3，注意这也是相当于规范化之后的数值
b = tf.Variable(np.float32(0.3))
a = tf.Variable(np.float32(0.3))

#这是定义主要的数学模型，模型来自于上面的公式
#注意这里必须使用tf的公式，这样的公式才是模型
#上面使用np的是直接计算，而不是定义模型
# TensorFlow的函数名基本就是完整英文，你应当能读懂
y_value = tf.multiply(x,a) + b

# 这里是代价函数，同我们文中所讲的唯一区别是用平方来取代求绝对值，
#目标都是为了得到一个正数值，功能完全相同，
#平方计算起来会更快更容易,这种方式也称为“方差“
loss = tf.reduce_mean(input_tensor=tf.square(y_value - y))
# TensorFlow内置的梯度下降算法，每步长0.5
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
# 代价函数值最小化的时候，代表求得解
train = optimizer.minimize(loss)

# 初始化所有变量，也就是上面定义的a/b两个变量
init = tf.compat.v1.global_variables_initializer()

#启动图
sess = tf.compat.v1.Session()
#真正的执行初始化变量，还是老话，上面只是定义模型，并没有真正开始执行
sess.run(init)

#重复梯度下降200次，每隔5次打印一次结果
for step in range(0, 200):
    sess.run(train) 
    if step % 5 == 0:
        print(step, sess.run(loss),sess.run(a), sess.run(b))
