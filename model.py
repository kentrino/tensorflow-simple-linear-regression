import tensorflow as tf

DIMENSION = 1


def weight_variable(name, shape):
    # with tf.device('/cpu:0'):
    # initializer = tf.truncated_normal_initializer(stddev=0.1)
    initializer = tf.constant_initializer(1.0)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float64)
    return var


def bias_variable(name, shape):
    initializer = tf.constant_initializer(1.0)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float64)
    return var


def inference(x):
    w = weight_variable("w", [DIMENSION])
    b = bias_variable("b", [DIMENSION])
    y = tf.add(tf.multiply(w, x), b)
    return y


def loss(y_inference, y):
    sub = tf.subtract(y_inference, y)
    square = tf.square(sub)
    mean = tf.reduce_mean(square)
    sqrt = tf.sqrt(mean)
    return sqrt
