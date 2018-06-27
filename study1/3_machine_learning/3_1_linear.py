'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf
import matplotlib.pyplot as plt

# 학습 데이터
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167, 7.042, 10.791, 5.313]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221, 2.827, 3.465, 1.65]
#n_samples = len(train_X)

# X와 Y의 입력값
X = tf.placeholder("float")
Y = tf.placeholder("float")

# 모델의 wright와 bias의 값을 0으로 초기화
W = tf.Variable(0., name="weight")
b = tf.Variable(0., name="bias")

# linear 모델을 생성
# y = ax + b
# prediction_Y = (Weight * X) + bias
# => 최적의 weight와 bias값을 찾는다
# pred = tf.add(tf.multiply(X, W), b)
pred = W*X + b

# Cost Function 설계 (with Mean squared error)
# => 각 직선에 대해 비용(데이터와 직선과의 거리)을 계산함
# (예측값 - 실제값) 제곱(pow) : 음수 제거, 거리 제곱으로 차이 극대화
# 미분에서 계산의 편의성을 위해 1/2의 곱을 더하는 경우가 있음
#cost = tf.reduce_sum(tf.pow(pred-Y, 2))/n_samples
cost = tf.reduce_mean(tf.square(pred-Y))

# Gradient descent Optimizer(학습)
# 미분을 통해서 해당 점의 기울기가 가장 작은 곳이 최적화의 포인트(learning_rate만큼의 단위로 실행)
# 지속적으로 기울기(미분)를 측정하여 W와 b를 수정
# W' = W - (cost함수의 미분값 * learning_rate:0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 학습 시작
with tf.Session() as sess:
    # 초기화 실행
    sess.run(tf.global_variables_initializer())

    #_cost = _W = _b = 0
    # 학습횟수(epoch:3000) 
    for epoch in range(3000):
        sess.run(optimizer, feed_dict={X: train_X, Y:train_Y})

        # 로그
        training_cost = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
        training_W = sess.run(W)
        training_b = sess.run(b)
        print(epoch, training_cost, [training_W, training_b])

        # cost가 더이상 변화가 없으면 종료
        #if _cost == training_cost and _W == training_W and _b == training_b:
        #    break;
        #_cost, _W, _b = (training_cost, training_W, training_b)

    print("학습완료! (cost : " + str(training_cost) + ")")


    # 새로운 값으로 테스트
    # iteration없고 optimizer없이, 테스트 데이터만 가지고 체크 
    # => cost 안에 이미 W와 b가 결정되었기 때문
    test_X = [6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1]
    test_Y = [1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03]

    testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
    print("테스트 완료! (cost : " + str(testing_cost) + ")")
    
    # 학습과 테스트 cost비교(절대값) 
    print("테스트와 학습의 cost차이 : ", abs(training_cost - testing_cost))

    # 화면표시
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, W.eval() * tf.constant(train_X).eval() + b.eval(), label='Fitted line')

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(test_X, W.eval() * tf.constant(test_X).eval() + b.eval(), label='fitted line(test)')

    plt.legend()
    plt.show()

    # 값 예측
    print("x가 3.3일때 : " + str(sess.run(pred, feed_dict={X: 3.3})))
