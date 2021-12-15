import numpy as np
import matplotlib.pyplot as plt

X = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
y = np.array([3.0, 3.5, 5.5, 6.0, 7.5, 8.0, 8.5, 9.0, 9.5, 9.9])

W = 0
b = 0

lrate = 0.01
epochs = 1000

n = float(len(X))

for i in range(epochs):
    y_pred = W*X + b
    dW = (2/n) * sum(X * (y_pred-y))
    db = (2/n) * sum(y_pred-y)
    W = W - lrate * dW
    b = b - lrate * db

print(W, b)

y_pred = W*X + b
plt.scatter(X, y)

plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color = 'red')
plt.show()