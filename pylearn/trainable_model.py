from .preprocess import InputData, InitialParameters, FeatureScaling
from .cost import rss


class TrainableModel:

    def __init__(self):
        self.max_iterations = 1000
        self.learning_rate = 0.5
        self.regularization_term = 0
        self.train_threshold = 0.01
        self.theta = []
        self.feature_scale = lambda _: 1
        self.reverse_feature_scale = lambda _: 1
        self.predict = lambda _: 1
        self.gradient_log = []

    def fit(self, X, y):
        mean, reversed_mean = FeatureScaling.get_mean_normalize(X)
        X, y = InputData.normalize(mean(X), y)

        cost, derivative = rss(X, y, self.hypothesis, self.regularization_term)
        theta = InitialParameters.random(len(X[0]))
        theta = self.train(derivative, theta)
        self.theta = theta
        self.feature_scale = mean
        self.reverse_feature_scale = reversed_mean

        def predictor(inp):
            return sum(x * t for x, t in zip([1] + list(mean(inp)), theta))
        self.predict = predictor

        return predictor

    # Gradient descent
    def train(self, derivative, theta):
        self.gradient_log = []
        last_derivative = derivative(theta)
        iteration = self.max_iterations
        current_error = sum(map(abs, last_derivative))
        while iteration and current_error > self.train_threshold:
            theta = theta - self.learning_rate * last_derivative
            last_derivative = derivative(theta)
            current_error = sum(map(abs, last_derivative))

            self.gradient_log.append(current_error)
            iteration -= 1

        return theta
