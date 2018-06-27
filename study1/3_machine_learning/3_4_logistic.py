'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 데이터
train_X = [0.3, 0.4, 9.5, 6.71, 8.93, 2.168, 9.779, 7.182, 8.59, 0.167, 7.042, 6.791, 8.313, 7.182, 8.59, 0.167, 7.042, 6.791, 8.313, 0.5, 0.4, 0.3, 0.3, 0.4, 7.5, 9.71, 7.93, 0.168, 8.779, 6.182, 7.59, 4.167, 6.042, 9.791, 7.313, 6.182, 6.59, 0.167, 6.042, 0.791, 7.313, 0.5, 1.4, 0.3]
train_Y = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

# X와 Y의 입력값
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 모델의 wright와 bias의 값을 random으로 초기화
W = tf.Variable(0., name="weight")
b = tf.Variable(0., name="bias")

# Logistic 모델을 생성
#pred = 1. / (1. + tf.exp(-(W*X + b)))
pred = tf.sigmoid(W*X + b) 

# Cost Function 설계 (with Cross Entropy)
cost = tf.reduce_mean(tf.reduce_sum(-Y * tf.log(pred) - (1 - Y) * tf.log(1 - pred)))

# Gradient descent Optimizer(학습)
# 미분을 통해서 해당 점의 기울기가 가장 작은 곳이 최적화의 포인트(learning_rate만큼의 단위로 실행)
# 지속적으로 기울기(미분)를 측정하여 W와 b를 수정
# W' = W - (cost함수의 미분값 * learning_rate:0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 학습 시작
with tf.Session() as sess:
    # 초기화 실행
    sess.run(tf.global_variables_initializer())

    # 학습횟수(epoch:1000) 
    for epoch in range(1000):
        sess.run(optimizer, feed_dict={X: train_X, Y:train_Y})

        # 로그
        training_cost = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        training_W = sess.run(W)
        training_b = sess.run(b)
        print(epoch, training_cost, [training_W, training_b])

    print("학습완료! (cost : " + str(training_cost) + ")")

    # 새로운 값으로 테스트
    # iteration없고 optimizer없이, 테스트 데이터만 가지고 체크 
    # => cost 안에 이미 W와 b가 결정되었기 때문
    test_X = [0.3, 0.4, 1.5, 7.71, 7.93, 0.168, 8.779, 9.182]
    test_Y = [0, 0, 0, 1, 1, 0, 1, 1]

    testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
    print("테스트 완료! (cost : " + str(testing_cost) + ")")
    
    # 학습과 테스트 cost비교(절대값) 
    print("테스트와 학습의 cost차이 : ", abs(training_cost - testing_cost))

    # 화면표시
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(test_X, test_Y, 'bo', label='Testing data')

    plt.legend()
    plt.show()

    # 값 예측
    print("x가 0.15일때 : " + str(sess.run(pred, feed_dict={X: [0.15]})))
    print("x가 8.5일때 : " + str(sess.run(pred, feed_dict={X: [8.5]})))
    print("x가 6.5일때 : " + str(sess.run(pred, feed_dict={X: [6.5]})))
    print("x가 0.9일때 : " + str(sess.run(pred, feed_dict={X: [0.9]})))
