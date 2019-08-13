import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot
import numpy as np
import tensorflow as tf

def get_fitness(** kwargs):
    red_tree = kwargs.get('red_tree')
    green_tree = kwargs.get('green_tree')
    blue_tree = kwargs.get('blue_tree')
    alpha_tree = kwargs.get('alpha_tree')
    x_size = kwargs.get('x_size')
    y_size = kwargs.get('y_size')
    current_individual = kwargs.get('current_individual')
    current_generation = kwargs.get('current_generation')
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with sess.graph.device("/gpu:1"):
            red_tensor = tf.cast(red_tree.get_tensor(x_size, y_size), tf.uint8)
            green_tensor = tf.cast(green_tree.get_tensor(x_size, y_size), tf.uint8)
            blue_tensor = tf.cast(blue_tree.get_tensor(x_size, y_size), tf.uint8)
            alpha_tensor = tf.cast(alpha_tree.get_tensor(x_size, y_size), tf.uint8)
            red_result = sess.run(red_tensor)
            green_result = sess.run(green_tensor)
            blue_result = sess.run(blue_tensor)
            alpha_result = sess.run(alpha_tensor)
            auxiliary_array = [red_result, green_result, blue_result, alpha_result]
            result = np.stack(auxiliary_array, axis=2)
            matplotlib.pyplot.imsave('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png', result)
    image_size = os.path.getsize('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png')
    return -image_size
