class TrainableModel:

    def __init__(self, params=[]):
        self.max_iterations = 1000
        self.learning_rate = 0.002
        self.regularization_term = 0
        self.train_threshold = 0.0001
        self.predict = lambda _: 1

    def train(self, derivative, theta):
        last_derivative = derivative(theta)
        iteration = self.max_iterations
        while iteration and abs(sum(last_derivative)) > self.train_threshold:
            theta = theta - self.learning_rate * last_derivative
            last_derivative = derivative(theta)
            iteration -= 1

        self.predict = lambda inp: sum(x * t for x, t in zip([1] + inp, theta))

        return theta
