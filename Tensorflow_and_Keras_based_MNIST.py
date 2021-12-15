import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='sigmoid', input_shape=(784,)))
model.add(tf.keras.layers.Dense(num_classes, activation='sigmoid'))

model.summary()

sgd = tf.keras.optimizers.SGD(lr=0.1)

model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print('테스트 손실값', score[0])
print('테스트 정확도', score[1])

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()