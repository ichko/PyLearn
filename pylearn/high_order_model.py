from .linear_model import LinearRegression, LogisticRegression


def power_map(powers):
    return (lambda data: data ** power for power in powers)


def identity_map(size):
    return [lambda x, id: x[id] for _ in range(size)]


def row_apply_map(mapper, x_row):
    return [transform(x_row, i) for i, transform in enumerate(mapper)]


class PolynomialRegression(LinearRegression):

    def fit(self, X, y, mapper=None):
        mapper = identity_map(len(X[0])) if mapper is None else mapper
        X = [row_apply_map(mapper, x_row) for x_row in X]
        predict = super(LinearRegression, self).fit(X, y)
        return lambda inp: predict(row_apply_map(mapper, inp))


class PolynomialLogisticRegression(LogisticRegression):

    def fit(self, X, y, mapper=None):
        mapper = identity_map(len(X[0])) if mapper is None else mapper
        X = [row_apply_map(mapper, x_row) for x_row in X]
        super(PolynomialLogisticRegression, self).fit(X, y)

        base_unthresholded = self.unthresholded
        self.unthresholded = lambda inp: base_unthresholded(
                             row_apply_map(mapper, inp))

        return lambda inp: 1 if self.unthresholded(inp) > 0 else 0
