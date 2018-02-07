# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def _generate_data(config):
    x, y = config.generate(config.data_size, config.dimension)
    if not tf.gfile.Exists(config.data_dir):
        tf.gfile.MakeDirs(config.data_dir)
    data_path = config.data_dir + "/" + config.data_file
    np.savez(data_path, x=x, y=y)
    return x, y


def get_data(config):
    data_path = config.data_dir + "/" + config.data_file
    if tf.gfile.Exists(data_path):
        data = np.load(data_path)
        return data["x"], data["y"]
    return _generate_data(config)


def show_data(config):
    x, y = get_data(config)
    plt.scatter(x, y)
    plt.savefig(config.data_dir + "/" + "image.png")


def get_trainable_variable(name):
    for v in tf.trainable_variables():
        if v.name == name:
            return v
    raise Exception("No variable named %s" % name)


def show_all_variables():
    keys = tf.get_default_graph().get_all_collection_keys()
    for key in keys:
        print("--- %s" % key)
        print(tf.get_collection(key))
        print("\n")
