class TrainableModel:

    def __init__(self, params=[]):
        self.max_iterations = 400
        self.learning_rate = 0.01
        self.regularization_term = 0
        self.derivative_threshold = 0.001
        self.hypothesis = lambda _: 1

    def train(self):
        raise NotImplementedError()
