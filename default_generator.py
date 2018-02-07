import numpy as np

SIZE = 10000


def _correct_model(x):
    return x * 3 + 2


def generate():
    r = np.float64(np.random.rand(SIZE))
    xs = np.multiply(r, 100) - 50
    ys = _correct_model(xs)
    return xs, ys
