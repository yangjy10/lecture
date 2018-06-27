'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

#그래프 생성
g = tf.Graph()
with g.as_default() as base_g:
    with base_g.name_scope("g1") as scope:
        # 상수를 생성하여 정의된 그래프에 추가 (상수 명령값을 출력하는 텐서를 하나 만든다)
        x = tf.constant(10, name="X")
        y = tf.constant(5, name="Y")
        x_y = tf.add(x, y, name="ADD")

# 세션 시작
with tf.Session(graph=g) as sess:
    # 그래프 계산 명령 실행
    print(sess.run(x_y))

