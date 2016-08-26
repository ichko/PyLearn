"""Module containing implementation of neural network multiclass classifier
with sigmoid neurons using batch gradient descent and backpropagation.

"""


import random

import numpy as np

from .math import sigmoid, sigmoid_prime
from .preprocess import InitialParameters


class NeuralNetwork:
    """NeuralNetwork class implements sigmoid neurons and `learns` the parameters
    fitting the data with the backpropagation algorithm.

    """

    def __init__(self, sizes, learning_rate=2,
                 initial_parameters=InitialParameters.random_matrix):
        """Initializing the parameters of the model.
        Setting initial random values of the weights and biases of the neural
        network.

        Play with the max_iteration, batch_size and learning_rate values
        for optimal performance and optimal fit.

        """
        self.learning_rate = learning_rate
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [initial_parameters(1, y)[0] for y in sizes[1:]]
        self.weights = [initial_parameters(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        self.weighted_layer = []
        self.activations = []
        self.epoch_end_notifier = lambda _, __: 1
        self.max_iteration = 15
        self.batch_size = 10
        self.batch_preprocess = random.shuffle

    def cost(self, x, y):
        """Cost is computed as the differences of squared which are
        then summed between the hypothesised data and the actual data.

        """
        hypothesis = self._forward(x)
        return sum((y - hypothesis) ** 2) / 2

    def fit(self, X_data, y_data):
        """Method initializing the process of batch gradient descent.
        The data is split on even number of sets batches and then
        gradient descent is performed over those batches.
        This is dont multiple times. Ideally until convergence.

        """
        training_data = [(x, y) for x, y in zip(X_data, y_data)]
        training_data_len = len(training_data)

        for i in range(self.max_iteration):
            self.batch_preprocess(training_data)
            batches = [training_data[k:k + self.batch_size] for k
                       in range(0, training_data_len, self.batch_size)]
            for batch in batches:
                b_grad, w_grad = self._batch_gradient(batch)
                self.biases = [b - (self.learning_rate / len(batch)) * bg
                               for b, bg in zip(self.biases, b_grad)]
                self.weights = [w - (self.learning_rate / len(batch)) * wg
                                for w, wg in zip(self.weights, w_grad)]
            self.epoch_end_notifier(self, i)

        return self.predict

    def predict(self, x):
        """Feed forward the network with the x vector and predict the
        maximal value from the output vector.

        """
        hypothesis = self._forward(x)
        winner = max(hypothesis)
        return [1 if h == winner else 0 for h in hypothesis]

    def test_network(self, X_test, y_test):
        """Method returning the number of correct guesses from the test data.

        """
        return sum(int(y == self.predict(x))
                   for x, y in zip(X_test, y_test))

    def batch_cost(self, X_data, y_data):
        """Method returning the sum of the cost of the model over the test data.

        """
        return sum(self.cost(x, y) for x, y in zip(X_data, y_data))

    def _forward(self, a):
        """Calculating the output of the network based on the current
        parameters and the input `a`.

        """
        a = np.array(a)
        self.weighted_layer, self.activations = [], [a]
        for w, b in zip(self.weights, self.biases):
            z = w.dot(a) + b
            a = sigmoid(z)
            self.weighted_layer.append(z)
            self.activations.append(a)

        return a

    def _batch_gradient(self, batch):
        """Method summing the gradient for each value in the batch."""
        b_grad = [np.zeros(b.shape) for b in self.biases]
        w_grad = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:
            single_b_grad, single_w_grad = self._cost_prime(x, y)
            b_grad = [b + bg for b, bg in zip(b_grad, single_b_grad)]
            w_grad = [w + wg for w, wg in zip(w_grad, single_w_grad)]

        return b_grad, w_grad

    def _cost_prime(self, x, y):
        """Running the backpropagation algorithm for single x, y input.
        The result is a tuple containing the gradient for the weights
        and the gradient for the biases.

        """
        x, y = np.array(x), np.array(y)
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        weights_gradient = [np.zeros(w.shape) for w in self.weights]
        hypothesis = self._forward(x)

        bias_gradient[-1] = (hypothesis - y) * \
            sigmoid_prime(self.weighted_layer[-1])
        weights_gradient[-1] = np.array([bias_gradient[-1]]).T.dot(
            np.array([self.activations[-2]]))

        for l in range(2, self.num_layers):
            bias_gradient[-l] = self.weights[-l + 1].T.dot(
                bias_gradient[-l + 1]) * sigmoid_prime(self.weighted_layer[-l])
            weights_gradient[-l] = np.array([bias_gradient[-l]]).T.dot(
                np.array([self.activations[-l - 1]]))

        return bias_gradient, weights_gradient
