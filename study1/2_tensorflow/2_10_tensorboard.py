'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

with tf.name_scope('op'):
    a = tf.constant(2, name="a")
    b = tf.constant(1, name="b")

with tf.name_scope('input'):
    x = tf.placeholder(tf.int32, name="X")
    y = tf.Variable(0, tf.int32, name="Y")

y = a * x + b

tf.summary.scalar('x', x)
tf.summary.scalar('y', y)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # TensorBoard 그래프용 로그 기록
    writer = tf.summary.FileWriter("/tmp/tensorboard", sess.graph)

    # summary 정보를 파일로 저장
    merged = tf.summary.merge_all()

    for i in range(10):
        summary, t = sess.run([merged, y], feed_dict={x: i})
        writer.add_summary(summary, i)

    writer.close()

