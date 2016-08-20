import numpy as np


class Math:

    @classmethod
    def sigmoid_prime(cls, x):
        s = cls.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class Network:

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.array(np.random.randn(1, y))[0] for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.weighted_layer = []
        self.activations = []

    def forward(self, a):
        a = np.array(a)
        self.weighted_layer, self.activations = [], [a]
        for w, b in zip(self.weights, self.biases):
            z = w.dot(a) + b
            a = Math.sigmoid(z)
            self.weighted_layer.append(z)
            self.activations.append(a)

        return a

    def cost(self, X, y):
        hypothesis = self.forward(X)
        return sum((y - hypothesis) ** 2) / 2

    def cost_prime(self, x, y):
        x, y = np.array(x), np.array(y)
        dJdb = [np.zeros(b.shape) for b in self.biases]
        dJdW = [np.zeros(w.shape) for w in self.weights]
        hypothesis = self.forward(x)

        dJdb[-1] = (hypothesis - y) * \
            Math.sigmoid_prime(self.weighted_layer[-1])
        dJdW[-1] = np.array([dJdb[-1]]).T.dot(
            np.array([self.activations[-2]]))

        for l in range(2, self.num_layers):
            dJdb[-l] = self.weights[-l + 1].T.dot(dJdb[-l + 1]) * \
                Math.sigmoid_prime(self.weighted_layer[-l])
            dJdW[-l] = np.array([dJdb[-l]]).T.dot(
                np.array([self.activations[-l - 1]]))

        return dJdW, dJdb


net = Network([2, 3, 2])
print('forward:', net.forward([2, 3]))

w_grad, b_grad = net.cost_prime([1, 2], [1, 3])
print('dJdW:', w_grad, '\ndJdb:', b_grad)

cost1 = net.cost([1, 2], [1, 2])
scalar = 3
net.weights[0] -= scalar * w_grad[0]
net.weights[1] -= scalar * w_grad[1]
cost2 = net.cost([1, 2], [1, 2])

print(cost1, cost2)
