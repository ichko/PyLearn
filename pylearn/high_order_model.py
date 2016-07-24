from .linear_model import LinearRegression


def power_map(powers):
    return (lambda data: data ** power for power in powers)


def identity_map(size):
    return [lambda x: x for _ in range(size)]


class PolynomialRegression(LinearRegression):

    @staticmethod
    def row_apply_map(mapper, x_row):
        return [transform(x) for transform, x in zip(mapper, x_row)]

    def fit(self, X, y, mapper=None):
        mapper = identity_map(len(X[0])) if mapper is None else mapper
        X = [self.row_apply_map(mapper, x_row) for x_row in X]
        predict = super(LinearRegression, self).fit(X, y)
        return lambda inp: predict(self.row_apply_map(mapper, inp))
