'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import datas.mnist_data as mnist

# 학습 데이터(MNIST)
(train_labels, train_images) = mnist.get_data('./datas/', 'train')

# X(이미지)와 Y(숫자값)의 입력값
X = tf.placeholder(tf.float32, [None, 28*28]) # 무한대 x 784(28*28) 행렬
Y = tf.placeholder(tf.int64, [None]) # 무한대 x 1 행렬

# CNN Model
def model(input_X): 
    x_reshape = tf.reshape(X, [-1, 28, 28, 1])

    # Conv 레이어1
    W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1)) # Filter가 Weight 역활을 함
    b_conv1 = tf.Variable(tf.zeros([20]))
    h_conv1 = tf.nn.conv2d(x_reshape, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
    fmap_conv1 = tf.nn.tanh(h_conv1) # -> Feature(Activation) Map 생성

    # Pooling(Max) 레이어1
    h_pool1 = tf.nn.max_pool(fmap_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fully-connected 레이어 1
    W_fc1 = tf.Variable(tf.truncated_normal([20 * 14 * 14, 500], stddev=0.1))
    b_fc1 = tf.Variable(tf.zeros([500]))
    h_pool1_flat = tf.reshape(h_pool1, [-1, 20 * 14 * 14])
    h_fc1 = tf.nn.tanh(tf.matmul(h_pool1_flat, W_fc1) + b_fc1) 

    # 출력(fully connected) 레이어 2 (10개 출력)
    class_num = 10
    W_fc2 = tf.Variable(tf.truncated_normal([500, class_num], stddev=0.1))
    b_fc2 = tf.Variable(tf.zeros([class_num]))
    pred = tf.matmul(h_fc1, W_fc2) + b_fc2

    return pred

pred = model(X)

# Cost Function 설계 (Cross Entropy)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=pred)
cost = tf.reduce_mean(cross_entropy)

# Gradient descent Optimizer(학습)
# 미분을 통해서 해당 점의 기울기가 가장 작은 곳이 최적화의 포인트(learning_rate만큼의 단위로 실행)
# 지속적으로 기울기(미분)를 측정하여 W와 b를 수정
# W' = W - (cost함수의 미분값 * learning_rate:0.01)
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

# 학습 시작
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    index_in_epoch = 0
    # 학습횟수(epoch:60) -> 총 60000개 
    for epoch in range(1, 60):
        start = index_in_epoch
        index_in_epoch += 1000 # 배치 1000개
        end = index_in_epoch

        X_images = np.reshape(train_images[start:end], [-1, 28*28])
        Y_labels = train_labels[start:end]

        sess.run(optimizer, feed_dict={X: X_images , Y: Y_labels})
        
        # 로그
        training_cost = sess.run(cost, feed_dict={X: X_images , Y: Y_labels})
        print(epoch, training_cost)
        #print(epoch, training_cost, [W.eval(), b.eval()])

    print("학습완료! (cost : " + str(training_cost) + ")")

    # 테스트 데이터로 테스트(총 10000개)
    # iteration없고 optimizer없이, 테스트 데이터만 가지고 체크 
    # => cost 안에 이미 W와 b가 결정되었기 때문
    (test_labels, test_images) = mnist.get_data('./datas/', 'test')

    test_X = np.reshape(test_images, [-1, 28*28])
    test_Y = test_labels

    testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y})
    print("테스트 완료! (cost : " + str(testing_cost) + ")")
    
    # 학습과 테스트 cost비교(절대값) 
    print("테스트와 학습의 cost차이 : ", abs(training_cost - testing_cost))

    # 값 예측 (10개)
    for i in range(20): 
        x_test = np.reshape(train_images[i], [-1, 28*28])
        arr_data = sess.run(pred, feed_dict={X: x_test})

        pred_val = tf.argmax(arr_data, 1)
        real_val = train_labels[i]

        print("예측값: " + str(pred_val.eval()) + " / 실제값" + str(real_val) + " => " + str(tf.equal(pred_val, real_val).eval()))
