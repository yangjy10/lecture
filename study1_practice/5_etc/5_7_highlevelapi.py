import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터
x_data = [2, 4, 6, 8, 10, 20, 1, 2, 3, 4, 5]
y_data = [4, 8, 12, 16, 20, 40, 2, 4, 5, 8, 10 ]

column_x = tf.feature_column.numeric_column("x", dtype=tf.float32)
columns = [column_x]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=columns, optimizer="Adam")

# 학습
input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x = {"x":np.array(x_data[:6], dtype=np.float32)},
    y = np.array(y_data[:6], dtype=np.float32),
    num_epochs=10, batch_size=5, shuffle=True
)
estimator.fit(input_fn = input_fn_train,steps=5000)

# 검증
input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    x = {"x":np.array(x_data[7:], dtype=np.float32)},
    y = np.array(y_data[7:], dtype=np.float32),
    num_epochs=10, batch_size=5, shuffle=True
)
estimator.evaluate(input_fn = input_fn_eval,steps=10)

#예측
input_fn_predict = tf.estimator.inputs.numpy_input_fn(
    x = {"x":np.array([15,20,25,30], dtype=np.float32)},
    num_epochs=1, shuffle=True
)
result = list(estimator.predict(input_fn = input_fn_predict))

print(result)
