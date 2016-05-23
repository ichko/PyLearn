from .preprocess import InputData, InitialParameters
from .cost import rss
from .trainable_model import TrainableModel


class LinearRegression(TrainableModel):

    def fit(self, X, y):
        X, y = InputData.normalize(X, y)
        cost, derivative = rss(X, y, self.regularization_term)
        theta = InitialParameters.ones(len(y))
        return self.train(derivative, theta)

    def train(self, derivative, theta):
        last_derivative = derivative(theta)
        iteration = self.max_iterations
        while iteration and sum(last_derivative) < self.derivative_threshold:
            last_derivative = derivative(theta)
            theta = theta - self.learning_rate * last_derivative
            iteration -= 1

        return theta
