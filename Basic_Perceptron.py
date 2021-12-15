from sklearn.linear_model import Perceptron

X = [[0, 0],[0,1],[1,0],[1,1]]
y = [0, 0, 0, 1]

clf = Perceptron(tol=1e-3, random_state=0)

clf.fit(X, y)

print(clf.predict(X))

def calculate(input):
    global weights
    global bias
    activation = bias
    for i in range(2):
        activation += weights[i] * input[i]
    if activation >= 0.0:
        return 1.0
    else:
        return 0.0

def train_weights(X, y, l_rate, n_epoch):
    global weights
    global bias
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row, target in zip(X, y):
            actual = calculate(row)
            error = target - actual
            bias  = bias + l_rate * error
            sum_error += error**2
            for i in range(2):
                weights[i] = weights[i] + l_rate * error * row[i]
            print(weights, bias)
        print('에포크 번호=%d, 학습률=%.3f, 오류=%.3f' % (epoch, l_rate, sum_error))
    return weights

# X = [[0, 0],[0,1],[1,0],[1,1]]
# y = [0, 0, 0, 1]

# weights = [0.0, 0.0]
# bias = 0.0

# l_rate = 0.1
# n_epoch = 5
# weights = train_weights(X, y, l_rate, n_epoch)
# print(weights, bias)

X = [[0, 0],[0,1],[1,0],[1,1]]
y = [0, 1, 1, 0]

weights = [0.0, 0.0]
bias = 0.0

l_rate = 0.1
n_epoch = 100
weights = train_weights(X, y, l_rate, n_epoch)
print(weights, bias)