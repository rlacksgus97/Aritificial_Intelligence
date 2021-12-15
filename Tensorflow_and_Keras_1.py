import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = data = np.linspace(1, 2, 200)
y = X*4 + np.random.randn(200) * 0.3

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1, activation='linear'))
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
model.fit(X, y, batch_size=1, epochs=60)

predict = model.predict(data)

plt.plot(data, predict, 'b', data, y, 'k.')
plt.show()