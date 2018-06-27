'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

x = tf.placeholder(tf.int32, shape=(2, 2))
y = tf.matmul(x, x)

with tf.Session() as sess:
    # placeholder 입력 값 할당 전에 출력하면 에러 발생
    #print(sess.run(y))  

    matrix = [[1, 2],
              [1, 2]]
    print(sess.run(y, feed_dict={x: matrix})) # 성공적으로 출력
