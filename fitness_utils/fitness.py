import os

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot
import numpy as np
import tensorflow as tf

from .model_evaluator import train_model


def get_fitness(mode, ** kwargs):
    if mode == 'standard':
        individual_result = kwargs.get('individual_result')
        target_result = kwargs.get('target_result')
        depth = kwargs.get('depth')
        sum_absolute_error = 0
        for i in range(len(individual_result)):
            sum_absolute_error += abs(individual_result[i] - target_result[i])
        return sum_absolute_error + depth / 8
    if mode == 'keras':
        function_string = kwargs.get('function_string')
        current_individual = kwargs.get('current_individual')
        current_generation = kwargs.get('current_generation')
        value = train_model(current_generation, current_individual, function_string)
        print(value)
        return value
    if mode == 'image_gen':
        red_tree = kwargs.get('red_tree')
        green_tree = kwargs.get('green_tree')
        blue_tree = kwargs.get('blue_tree')
        alpha_tree = kwargs.get('alpha_tree')
        current_individual = kwargs.get('current_individual')
        current_generation = kwargs.get('current_generation')
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            with sess.graph.device("/gpu:1"):
                x_size = 255
                y_size = 255
                red_tensor = tf.cast(red_tree.get_tensor(x_size, y_size), tf.uint8)
                green_tensor = tf.cast(green_tree.get_tensor(x_size, y_size), tf.uint8)
                blue_tensor = tf.cast(blue_tree.get_tensor(x_size, y_size), tf.uint8)
                alpha_tensor = tf.cast(alpha_tree.get_tensor(x_size, y_size), tf.uint8)
                red_result = sess.run(red_tensor)
                green_result = sess.run(green_tensor)
                blue_result = sess.run(blue_tensor)
                alpha_result = sess.run(alpha_tensor)
                result = np.stack((red_result, green_result, blue_result, alpha_result), axis=2)
                matplotlib.pyplot.imsave('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png', result)
        image_size = os.path.getsize('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png')
        return -image_size
