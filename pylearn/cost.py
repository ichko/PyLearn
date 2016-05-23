def rss_cost(X, y, reg_term):
    training_set_size = len(y)

    def cost_function(theta):
        hypothesis = X.transpose().dot(theta)
        reg_sum = reg_term * sum(theta[1:] ** 2)
        result = (sum((hypothesis - y) ** 2) + reg_sum)
        return result / (2 * training_set_size)

    return cost_function


def rss_derivative(X, y, reg_term):
    training_set_size = len(y)

    def derivative(theta):
        hypothesis = X.dot(theta)
        gradient = X.transpose().dot(hypothesis - y) / training_set_size
        gradient[1:] += reg_term * theta[1:] / training_set_size
        return gradient

    return derivative


def rss(X, y, reg_term=0):
    return rss_cost(X, y, reg_term), rss_derivative(X, y, reg_term)
