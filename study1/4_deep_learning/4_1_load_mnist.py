'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

import tensorflow as tf
from matplotlib import pyplot as plt

import datas.mnist_data as mnist

# 학습 데이터(MNIST)
(train_labels, train_images) = mnist.get_data('./datas/', 'train')
(test_labels, test_images) = mnist.get_data('./datas/', 'test')

# 데이터 사이즈 출력
print('total train data : ' + str(train_labels.size))
print('total test data : ' + str(test_labels.size))

# 임의의 수 내용 출력
print(train_images[0])
print(train_labels[0])

print("----------------------")

# 10개 수 출력(with matplot)
print('label: %s' % (train_labels[0:10]))

for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(train_images[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
