import keras
print(keras.__version__)

from tensorflow import keras
from tensorflow.keras import layers
# 선형 유닛 1개로 구성된 네트워크 생성
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])

model.weights
model.summary()

w, b = model.weights

print()

import tensorflow as tf
x = tf.linspace(-1.0, 1.0, 3)
x = tf.reshape(x, (1, 3))
y = model.predict(x)

model = keras.Sequential([
    layers.Dense(units=5, input_shape=[3])
])


model = keras.Sequential([
    # 은닉층 (ReLU 활성화 함수 사용)
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # 출력층 (선형 활성화)
    layers.Dense(units=1),
])

model