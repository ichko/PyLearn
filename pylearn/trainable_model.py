class TrainableModel:

    def __init__(self, params=[]):
        self.max_iterations = 1000
        self.learning_rate = 0.002
        self.regularization_term = 0
        self.train_threshold = 0.0001
        self.predict = lambda _: 1

    def train(self):
        raise NotImplementedError()
