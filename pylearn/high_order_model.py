from .linear_model import LinearRegression, LogisticRegression


def _identity_map(size):
    return [lambda x, id: x[id] for _ in range(size)]


def _row_apply_map(mapper, x_row):
    return [transform(x_row, i) for i, transform in enumerate(mapper)]


class PolynomialRegression(LinearRegression):

    def fit(self, X, y, mapper=None):
        mapper = _identity_map(len(X[0])) if mapper is None else mapper
        X = [_row_apply_map(mapper, x_row) for x_row in X]
        predict = super(LinearRegression, self).fit(X, y)

        def predictor(inp, params=self.params):
            return predict(_row_apply_map(mapper, inp), params)

        self.predict = predictor

        return predictor


class PolynomialLogisticRegression(LogisticRegression):

    def fit(self, X, y, mapper=None):
        mapper = _identity_map(len(X[0])) if mapper is None else mapper
        X = [_row_apply_map(mapper, x_row) for x_row in X]
        super(PolynomialLogisticRegression, self).fit(X, y)

        base_predict = self.predict
        self.predict = lambda inp, params=self.params: base_predict(
                _row_apply_map(mapper, inp), params)

        return lambda inp: 1 if self.predict(inp) > 0 else 0
