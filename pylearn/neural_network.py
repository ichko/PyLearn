import random

import numpy as np

from .math import sigmoid, sigmoid_prime


class NeuralNetwork:

    def __init__(self, sizes, learning_rate=2):
        self.learning_rate = learning_rate
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.array(np.random.randn(1, y))[0] for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.weighted_layer = []
        self.activations = []
        self.epoch_end_notifier = lambda _, __: 1
        self.max_iteration = 15
        self.batch_size = 10

    def cost(self, x, y):
        hypothesis = self._forward(x)
        return sum((y - hypothesis) ** 2) / 2

    def fit(self, X_data, y_data):
        training_data = [(x, y) for x, y in zip(X_data, y_data)]
        training_data_len = len(training_data)

        for i in range(self.max_iteration):
            random.shuffle(training_data)
            batches = [training_data[k:k + self.batch_size] for k
                       in range(0, training_data_len, self.batch_size)]
            for batch in batches:
                b_grad, w_grad = self._batch_gradient(batch)
                self.biases = [b - (self.learning_rate / len(batch)) * bg
                               for b, bg in zip(self.biases, b_grad)]
                self.weights = [w - (self.learning_rate / len(batch)) * wg
                                for w, wg in zip(self.weights, w_grad)]
            self.epoch_end_notifier(self, i)

    def predict(self, x):
        hypothesis = self._forward(x)
        winner = max(hypothesis)
        return [1 if h == winner else 0 for h in hypothesis]

    def test_network(self, X_test, y_test):
        return sum(int(y == self.predict(x))
                   for x, y in zip(X_test, y_test))

    def batch_cost(self, X_data, y_data):
        return sum(self.cost(x, y) for x, y in zip(X_data, y_data))

    def _forward(self, a):
        a = np.array(a)
        self.weighted_layer, self.activations = [], [a]
        for w, b in zip(self.weights, self.biases):
            z = w.dot(a) + b
            a = sigmoid(z)
            self.weighted_layer.append(z)
            self.activations.append(a)

        return a

    def _batch_gradient(self, batch):
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            dJdb, dJdW = self._cost_prime(x, y)
            b_grad = [b + bg for b, bg in zip(b_grad, dJdb)]
            w_grad = [w + wg for w, wg in zip(w_grad, dJdW)]

        return b_grad, w_grad

    def _cost_prime(self, x, y):
        x, y = np.array(x), np.array(y)
        dJdb = [np.zeros(b.shape) for b in self.biases]
        dJdW = [np.zeros(w.shape) for w in self.weights]
        hypothesis = self._forward(x)

        dJdb[-1] = (hypothesis - y) * \
            sigmoid_prime(self.weighted_layer[-1])
        dJdW[-1] = np.array([dJdb[-1]]).T.dot(
            np.array([self.activations[-2]]))

        for l in range(2, self.num_layers):
            dJdb[-l] = self.weights[-l + 1].T.dot(dJdb[-l + 1]) * \
                sigmoid_prime(self.weighted_layer[-l])
            dJdW[-l] = np.array([dJdb[-l]]).T.dot(
                np.array([self.activations[-l - 1]]))

        return dJdb, dJdW
