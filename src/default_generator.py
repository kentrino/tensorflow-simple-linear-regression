import numpy as np


def _correct_model(x):
    return x * 3 + 2


def generate(size, _dimension):
    r = np.float64(np.random.rand(size))
    xs = np.multiply(r, 100) - 50
    ys = _correct_model(xs)
    return xs, ys
