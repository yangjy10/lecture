'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

x = tf.Variable(0) # 담을 데이터 변수

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, "/tmp/model.ckpt") # 로드

print(sess.run(x)) # 로드된 값을 출력해본다

