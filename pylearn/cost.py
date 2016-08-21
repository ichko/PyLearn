"""Module containing functions for computing the cost (error between)
two data sets and the derivative of this cost based on hypothesis function
and regularization term.

"""


def sum_squares_cost(hypothesis, reg_term):
    """Function returning a function for computing the sum of the
    squares of the difference between the data computed by the hypothesis
    over the X input data set and the y label data set.

    Args:
        hypothesis (function): Computing matrix of inputs to vector of outputs.
        reg_term (double): The regularization parameter used to compute
        the error.

    """
    def cost_function(params, X, y):
        training_set_size = len(y)
        predictions = hypothesis(X, params)
        reg_sum = reg_term * sum(params[1:] ** 2)
        result = (sum((predictions - y) ** 2) + reg_sum)
        return result / (2 * training_set_size)

    return cost_function


def sum_squares_derivative(hypothesis, reg_term):
    """Function returning a function for computing the derivative
    of the function witch is returned by cost_function(...).

    Args:
        hypothesis (function): Computing matrix of inputs to vector of outputs.
        reg_term (double): The regularization parameter used to compute
        the error.

    """
    def derivative(params, X, y):
        training_set_size = len(y)
        predictions = hypothesis(X, params)
        gradient = X.transpose().dot(predictions - y) / training_set_size
        gradient[1:] += reg_term * params[1:] / training_set_size
        return gradient

    return derivative


def sum_squares(hypothesis, reg_term=0):
    """Function returning tuple of functions computing the cost and the
    derivative of the cost of two datasets based on hypothesis function.

    Args:
        hypothesis (function): Computing matrix of inputs to vector of outputs.
        reg_term (double): The regularization parameter used to compute
        the error.

    """
    return (sum_squares_cost(hypothesis, reg_term),
            sum_squares_derivative(hypothesis, reg_term))
