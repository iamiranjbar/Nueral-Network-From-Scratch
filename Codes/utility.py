import numpy as np
import matplotlib.pyplot as plot
from itertools import product


def alphabetize(x,y):
    if x.get_name()>y.get_name():
        return 1
    return -1


def abs_mean(values):
    """Compute the mean of the absolute values a set of numbers.
    For computing the stopping condition for training neural nets"""
    return np.mean(np.abs(values))


def sigmoid_derivative(output):
    return output * (1 - output)


def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def check_derivatives(network):
    weights = network.weights
    performance_element = network.performance
    epsilon = (10 ** (-8))
    for weight in weights:
        initial_weight = weight.get_value()
        initial_performance = performance_element.output()
        weight_derivative = performance_element.dOutdX(weight)
        new_weight = initial_weight + epsilon
        weight.set_value(new_weight)
        network.clear_cache()
        new_performance = performance_element.output()
        finite_difference = (new_performance - initial_performance) / epsilon
        weight.set_value(initial_weight)
        network.clear_cache()
        if abs(finite_difference-weight_derivative) > 0.001:
            return False
    return True


def plot_decision_boundary(network, xmin, xmax, ymin, ymax):
    x_diff = xmax - xmin
    y_diff = ymax - ymin
    x_coords = [xmin+(x_diff/500)*index for index in range(501)]
    y_coords = [ymin+(y_diff/500)*index for index in range(501)]
    decision_points_x = []
    decision_points_y = []
    for x, y in product(x_coords, y_coords):
        network.inputs[0].set_value(x)
        network.inputs[1].set_value(y)
        network.clear_cache()
        result = network.output.output()
        if result < 0.5:
            decision_points_x.append(x)
            decision_points_y.append(y)
    plot.scatter(decision_points_x, decision_points_y, color='skyblue')
    plot.show()
