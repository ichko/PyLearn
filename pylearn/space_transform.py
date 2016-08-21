"""Module containing functions for generating lists of lambda
functions mapping vectors to higher dimensions.

"""


import numpy as np


def full_polynomial_mapper(power, space_size):
    """Function returning a map for vector with length space_size and all
    the degrees to certain power.

    Example:
        input: power = 2, space_size = 2
        output: [X1, X2, X1^2, X2^2, X1 * X2]

    """
    result = []
    for deg in range(power + 1)[1:]:
        result.extend(list(degree_lambda_mapper(deg, space_size)))
    return result


def degree_lambda_mapper(power, space_size, data=[]):
    """Function mapping the rows of degrees to lambda functions
    defined over vectors.

    Example:
        input: power = 2, space_size = 2
        output: [X1^2, X2^2, X1 * X2]

    """
    if(space_size > 1):
        for deg in range(power + 1):
            yield from degree_lambda_mapper(
                power - deg, space_size - 1, data + [deg])
    else:
        degs = data + [power]
        yield lambda x, _: np.prod([x[i] ** degs[i] for i in range(len(x))])


def degree_mapper(power, space_size, data=[]):
    """Function returning matrix with rows containing all the degrees of
    that vector with length of space_size can be mapped to.

    Example:
        input: power = 2, space_size = 2
        output: [[2, 0], [0, 2], [1, 1]]

    """
    if(space_size > 1):
        for deg in range(power + 1):
            yield from degree_mapper(power - deg, space_size - 1, data + [deg])
    else:
        yield data + [power]
