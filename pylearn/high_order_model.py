"""Module containing implementation of high order (polynomial) models.

"""


from .linear_model import LinearRegression, LogisticRegression


def _identity_map(size):
    """Function returning list of lambdas mapping vector to itself."""
    return [lambda x, id: x[id] for _ in range(size)]


def _row_apply_map(mapper, x_row):
    """Function applying mapper (list of lambdas) to vector of values."""
    return [transform(x_row, i) for i, transform in enumerate(mapper)]


class PolynomialRegression(LinearRegression):
    """Class approximating data with a complex (polynomial) function
    in higher dimension.

    """

    def fit(self, X, y, mapper=None):
        """Function fitting data by first applying mapper (list of lambdas)
        to the training data matrix and then linearly minimizing the cost
        function resulting in complex fit in the space of the real data.

        """
        mapper = _identity_map(len(X[0])) if mapper is None else mapper
        X = [_row_apply_map(mapper, x_row) for x_row in X]
        predict = super(LinearRegression, self).fit(X, y)

        def predictor(inp, params=self.params):
            return predict(_row_apply_map(mapper, inp), params)

        self.predict = predictor

        return predictor


class PolynomialLogisticRegression(LogisticRegression):
    """Class separating the data from two binary classes with complex
    complex decision boundary.

    """

    def fit(self, X, y, mapper=None):
        """Function separating two classes by first applying mapper (list of lambdas)
        to the training data matrix and then executing the linear logistic
        classifier resulting in complex decision boundary over the real data.

        """
        mapper = _identity_map(len(X[0])) if mapper is None else mapper
        X = [_row_apply_map(mapper, x_row) for x_row in X]
        super(PolynomialLogisticRegression, self).fit(X, y)

        base_predict = self.predict
        self.predict = lambda inp, params=self.params: base_predict(
                _row_apply_map(mapper, inp), params)

        return lambda inp: 1 if self.predict(inp) > 0 else 0
