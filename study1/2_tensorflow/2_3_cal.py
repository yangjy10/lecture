'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf

# 상수를 생성하여 기본 그래프에 추가 (상수 명령값을 출력하는 텐서를 하나 만든다)
x = tf.constant(10)
y = tf.constant(5)

# add 노드, 파이썬의 기본 ‘+ 연산자’에 tf.add(x, y)를 재정의 한 것
x_y = x + y 

# 세션 시작
with tf.Session() as sess:
    # 그래프 계산 명령 실행
    print(sess.run(x_y))

