'''
Created on 2018. 6. 15.

@author: eric.hong@aidentify.io
'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation

# 초기화
model = Sequential()
model.add(Dense(1, input_shape=(1,)))

# 훈련을 위해 모델 준비:
# optimiser(stochastic gradient descent), loss(mean squared error)와 를 설정
model.compile(optimizer='sgd', loss='mse')

# 훈련데이터 준비 (Y = 2X - 1)
xs = [1, 2, 3, 4, 5, 6, 7, 8]
ys = [1, 3, 5, 7, 9, 11, 13, 15]

# 데이터로 훈련
model.fit(xs, ys, epochs=100)

# 새로운 입력값으로 테스트
print(model.predict([2]))
