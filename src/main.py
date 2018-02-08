# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

from . import model
from .util import get_data, get_trainable_variable
from .config import default_config

tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size.")


def batch(np_data, batch_size):
    with tf.name_scope('input'):
        data = tf.constant(np_data, dtype=tf.float64)
        queue = tf.train.input_producer(data, shuffle=False)
        dequeue = queue.dequeue()
        batch_tensor = tf.train.batch([dequeue], batch_size=batch_size)
        # 1 is dimension of w and b
        # 当たり前だがこれをやらないとmodelの計算がめちゃめちゃになる
        return tf.reshape(batch_tensor, (batch_size, 1))


def main(_argv):
    batch_size = tf.app.flags.FLAGS.batch_size

    x_raw, y_raw = get_data(default_config)
    x = batch(x_raw, batch_size)
    y = batch(y_raw, batch_size)
    y_inferred = model.inference(x, default_config.dimension)
    loss = model.loss(y_inferred, y)
    global_step_tensor = tf.Variable(0, name='global_step', trainable=False)

    # TODO: wはこのlearning_rateで大丈夫だがbの収束が遅い
    # optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-4)
    train = optimizer.minimize(loss, global_step=global_step_tensor)

    w = get_trainable_variable("w:0")
    b = get_trainable_variable("b:0")

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                global_step = tf.train.global_step(sess, global_step_tensor)
                _, loss_value = sess.run([train, loss])
                if global_step % 100 == 0:
                    print("step: %d, loss: %s, w: %s, b: %s" % (global_step, loss_value, w.eval(), b.eval()))
        except KeyboardInterrupt:
            pass
        finally:
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run(main=main)
