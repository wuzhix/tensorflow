import tensorflow as tf
import numpy as np
import os


# disable "The TensorFlow library wasn't compiled to use ... instructions" warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 用 NumPy 随机生成 2行100列 个数据
x_data = np.float32(np.random.rand(2, 100))
# np.dot 矩阵乘法 [1行,2列] * [2行,100列] = [1行,100列]
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 变量初始化为1行0列，值为0
b = tf.Variable(tf.zeros([1]))
# 变量初始化为1行2列，值为-1.0到1.0的随机数
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
# tf.matmul矩阵乘法 [1行,2列] * [2行,100列] = [1行,100列]
y = tf.matmul(W, x_data) + b

# 计算损失loss，tf.reduce_mean取所有数据的平均值，tf.square平方
loss = tf.reduce_mean(tf.square(y - y_data))
# 梯度下降算法，梯度下降值范围(0,1)，可以更换其他值测试
optimizer = tf.train.GradientDescentOptimizer(0.5)
# minimize添加操作节点，用于最小化loss，并更新var_list
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
with tf.Session() as sess:
    sess.run(init)

    # 拟合平面
    for step in range(201):
        # 开始训练
        sess.run(train)
        if step % 20 == 0:
            # 每训练20次输出一次训练结果
            print(step, sess.run(W), sess.run(b))
