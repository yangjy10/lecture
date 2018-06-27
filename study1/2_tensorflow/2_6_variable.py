'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

w = tf.Variable(tf.random_normal([3, 2]), name="W")
b = tf.Variable(w.initialized_value() + 3, name="B")

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print(sess.run(w))
    print(sess.run(b))
