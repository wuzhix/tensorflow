import tensorflow as tf
import os


# disable "The TensorFlow library wasn't compiled to use ... instructions" warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Save to file
# remember to define the same dtype and shape when restore
W = tf.Variable([[1, 2, 3], [3, 4, 5]], dtype=tf.float32, name='weights')
b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, "./save_net.ckpt")
    print("Save to path: ", save_path)
