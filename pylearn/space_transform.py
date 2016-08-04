import numpy as np


def full_polynomial_mapper(power, space_size):
    result = []
    for deg in range(power + 1)[1:]:
        result.extend(list(degree_lambda_mapper(deg, space_size)))
    return result


def degree_lambda_mapper(power, space_size, data=[]):
    if(space_size > 1):
        for deg in range(power + 1):
            yield from degree_lambda_mapper(
                power - deg, space_size - 1, data + [deg])
    else:
        degs = data + [power]
        yield lambda x, _: np.prod([x[i] ** degs[i] for i in range(len(x))])


def degree_mapper(power, space_size, data=[]):
    if(space_size > 1):
        for deg in range(power + 1):
            yield from degree_mapper(power - deg, space_size - 1, data + [deg])
    else:
        yield data + [power]
