def rss_cost(hypothesis, reg_term):
    def cost_function(params, X, y):
        training_set_size = len(y)
        predictions = hypothesis(X, params)
        reg_sum = reg_term * sum(params[1:] ** 2)
        result = (sum((predictions - y) ** 2) + reg_sum)
        return result / (2 * training_set_size)

    return cost_function


def rss_derivative(hypothesis, reg_term):
    def derivative(params, X, y):
        training_set_size = len(y)
        predictions = hypothesis(X, params)
        gradient = X.transpose().dot(predictions - y) / training_set_size
        gradient[1:] += reg_term * params[1:] / training_set_size
        return gradient

    return derivative


def rss(hypothesis, reg_term=0):
    return (rss_cost(hypothesis, reg_term),
            rss_derivative(hypothesis, reg_term))
