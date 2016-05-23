from .preprocess import InputData, InitialParameters
from .cost import rss
from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def fit(self, X, y):
        X, y = InputData.normalize(X, y)
        cost, derivative = rss(X, y, self.regularization_term)
        theta = InitialParameters.ones(len(X[0]))
        return self.train(derivative, theta)

    def train(self, derivative, theta):
        last_derivative = derivative(theta)
        iteration = self.max_iterations
        while iteration and abs(sum(last_derivative)) > self.train_threshold:
            last_derivative = derivative(theta)
            theta = theta - self.learning_rate * last_derivative
            iteration -= 1

        self.predict = lambda inp: sum(x * t for x, t in zip([1] + inp, theta))

        return theta
