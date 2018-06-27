'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

x = tf.Variable(2)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

sess = tf.Session()
sess.run(init_op)

save_path = saver.save(sess, "/tmp/model.ckpt")
