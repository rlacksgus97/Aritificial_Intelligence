import numpy as np

def actf(x):
    return 1/(1+np.exp(-x))

def actf_deriv(x):
    return x*(1-x)

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

y = np.array([[0], [1], [1], [0]])

np.random.seed(5)

inputs = 3
hiddens = 6
outputs = 1

weight0 = 2*np.random.random((inputs, hiddens))-1
weight1 = 2*np.random.random((hiddens, outputs))-1

for i in range(100000):

    layer0 = X
    net1 = np.dot(layer0, weight0)
    layer1 = actf(net1)
    layer1[:,-1] = 1.0
    net2 = np.dot(layer1, weight1)
    layer2 = actf(net2)

    layer2_error = layer2-y

    layer2_delta = layer2_error*actf_deriv(layer2)

    layer1_error = np.dot(layer2_delta, weight1.T)

    layer1_delta = layer1_error*actf_deriv(layer1)

    weight1 += -0.2*np.dot(layer1.T, layer2_delta)

    weight0 += -0.2*np.dot(layer0.T, layer1_delta)

print(layer2)