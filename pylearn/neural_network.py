import numpy as np

import random
import sys


class Math:

    @classmethod
    def sigmoid_prime(cls, x):
        s = cls.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class Network:

    def __init__(self, sizes, learning_rate=3):
        self.learning_rate = learning_rate
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

    def cost(self, x, y):
        hypothesis = self.forward(x)
        return sum((y - hypothesis) ** 2) / 2

    def train_network(self, X_data, y_data, test_data,
                      max_iter=15, batch_size=10):
        training_data = [(x, y) for x, y in zip(X_data, y_data)]
        training_data_len = len(training_data)

        for i in range(max_iter):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_size] for k
                       in range(0, training_data_len, batch_size)]
            for batch in batches:
                b_grad, w_grad = self.batch_gradient(batch)
                self.biases = [b - (self.learning_rate / len(batch)) * bg
                               for b, bg in zip(self.biases, b_grad)]
                self.weights = [w - (self.learning_rate / len(batch)) * wg 
                                for w, wg in zip(self.weights, w_grad)]
            if test_data:
                print("Epoch {0}: {1} / {2}, cost: {3}".format(
                    i, self.test_network(test_data[0], test_data[1]),
                    len(test_data[0]),
                    self.batch_cost(test_data[0], test_data[1])))
            else:
                print("Epoch {0} complete".format(i))

    def batch_gradient(self, batch):
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            dJdb, dJdW = self.cost_prime(x, y)
            b_grad = [b + bg for b, bg in zip(b_grad, dJdb)]
            w_grad = [w + wg for w, wg in zip(w_grad, dJdW)]

        return b_grad, w_grad

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

        return dJdb, dJdW

    def threshold_prediction(self, x):
        hypothesis = self.forward(x)
        winner = max(hypothesis)
        return [1 if h == winner else 0 for h in hypothesis]

    def test_network(self, X_test, y_test):
        return sum(int(y == self.threshold_prediction(x))
                   for x, y in zip(X_test, y_test))

    def batch_cost(self, X_data, y_data):
        return sum(self.cost(x, y) for x, y in zip(X_data, y_data))


'''
net = Network([2, 3, 2])
print('forward:', net.forward([2, 3]))

b_grad, w_grad = net.cost_prime([1, 2], [1, 3])
print('dJdW:', w_grad, '\ndJdb:', b_grad)

cost1 = net.cost([1, 2], [0, 1])
forward1 = net.forward([1, 2])
net.train_network([[1, 2], [1, 2], [1, 2]], [[0, 1], [0, 1], [0, 1]])
forward2 = net.forward([1, 2])
cost2 = net.cost([1, 2], [0, 1])

print(cost1, cost2)
print(forward1, forward2)
'''
