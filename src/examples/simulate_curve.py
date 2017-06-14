import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


# disable "The TensorFlow library wasn't compiled to use ... instructions" warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def add_layer(inputs, in_size, out_size, activatuib_funaction=None):
    # tf.Variable初始化变量 tf.random_normal正太分布随机数，均值mean,标准差stddev
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # tf.Variable初始化变量 tf.zeros生成值为0的数据
    biases = tf.Variable(tf.zeros([1, out_size]))+0.1
    # tf.matmul矩阵乘法
    wx_plus_b = tf.matmul(inputs, weights)+biases

    # activatuib_funaction激活函数
    if activatuib_funaction is None:
        outputs = wx_plus_b
    else:
        outputs = activatuib_funaction(wx_plus_b)
    return outputs

# np.linspace在指定的间隔内返回均匀间隔的数字，这里是返回起始-1，结束1的300个值。
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# 给出均值为mean，标准差为stdev的高斯随机数（场），当size赋值时，例如：size=100，表示返回100个高斯随机数。
noise = np.random.normal(0, 0.05, x_data.shape)
# np.square平方
y_data = np.square(x_data)-0.5+noise

# tf.placeholder占位符
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# tf.nn.relu计算激活函数relu，即max(features, 0)。
l1 = add_layer(xs, 1, 10, activatuib_funaction=tf.nn.relu)
predition = add_layer(l1, 10, 1, activatuib_funaction=None)

# tf.reduce_mean取所有数据的平均值 tf.square平方
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition), reduction_indices=[1]))
# GradientDescentOptimizer梯度下降算法 minimize添加操作节点，用于最小化loss，并更新var_list
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # figure新建绘画窗口,独立显示绘画的图片
    fig = plt.figure()
    # add_subplot将画布分割成1行1列，图像画在从左到右从上到下的第1块
    ax = fig.add_subplot(1, 1, 1)
    # scatter绘制散点
    ax.scatter(x_data, y_data)
    # 显示figure
    plt.show(block=False)

    sess.run(init)
    for train in range(1000):
        # feed_dict给占位符赋值
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if train % 50 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            print(train, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            predition_value = sess.run(predition, feed_dict={xs: x_data})
            # plot绘制属性
            lines = ax.plot(x_data, predition_value, 'r-', lw=5)
            # pause暂停0.1s
            plt.pause(0.1)
