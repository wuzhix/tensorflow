import tensorflow as tf
import os


# disable "The TensorFlow library wasn't compiled to use ... instructions" warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# －创建一个变量, 初始化为标量 0.  初始化定义初值
state = tf.Variable(0, name="counter")

# 创建一个 op, 其作用是使 state 增加 1
one = tf.constant(1)
# new_value = state + one
new_value = tf.add(state, one)
# state = new_value
update = tf.assign(state, new_value)

# 初始化variable
init_op = tf.global_variables_initializer()

# 启动默认图, 运行 op
with tf.Session() as sess:
    # 运行 'init' op
    sess.run(init_op)

    '''
    打印 'state' 的初始值
    取回操作的输出内容, 可以在使用 Session 对象的 run() 调用 执行图时, 
    传入一些 tensor, 这些 tensor 会帮助你取回结果. 
    此处只取回了单个节点 state，
    也可以在运行一次 op 时一起取回多个 tensor: 
    result = sess.run([mul, intermed])
    '''
    print(sess.run(state))

    # 运行 op, 更新 'state', 并打印 'state'
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
