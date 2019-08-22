import os

import numpy as np
import tensorflow as tf
import imageio

def get_fitness(** kwargs):
    red_tree = kwargs.get('red_tree')
    green_tree = kwargs.get('green_tree')
    blue_tree = kwargs.get('blue_tree')
    alpha_tree = kwargs.get('alpha_tree')
    x_size = kwargs.get('x_size')
    y_size = kwargs.get('y_size')
    current_individual = kwargs.get('current_individual')
    current_generation = kwargs.get('current_generation')
    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        with sess.graph.device("/gpu:1"):
            print(red_tree.get_string())
            red_tensor = tf.cast(red_tree.get_tensor(x_size, y_size), tf.uint8)
            print(green_tree.get_string())
            green_tensor = tf.cast(green_tree.get_tensor(x_size, y_size), tf.uint8)
            print(blue_tree.get_string())
            blue_tensor = tf.cast(blue_tree.get_tensor(x_size, y_size), tf.uint8)
            print(alpha_tree.get_string())
            alpha_tensor = tf.cast(alpha_tree.get_tensor(x_size, y_size), tf.uint8)
            red_result = sess.run(red_tensor)
            green_result = sess.run(green_tensor)
            blue_result = sess.run(blue_tensor)
            alpha_result = sess.run(alpha_tensor)
            auxiliary_array = [red_result, green_result, blue_result]
            result = np.stack(auxiliary_array, axis=2)
            matplotlib.pyplot.imsave('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png', result)
    image_size = os.path.getsize('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png')
    return -image_size
def get_image_fitness(** kwargs):
    red_tree = kwargs.get('red_tree')
    green_tree = kwargs.get('green_tree')
    blue_tree = kwargs.get('blue_tree')
    alpha_tree = kwargs.get('alpha_tree')
    x_size = kwargs.get('x_size')
    y_size = kwargs.get('y_size')
    current_individual = kwargs.get('current_individual')
    current_generation = kwargs.get('current_generation')
    image_to_fit = kwargs.get('image_to_fit')
    best_fit = kwargs.get('best_fit')
    tf.compat.v1.reset_default_graph()
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        red_tensor = tf.cast(red_tree.get_tensor(x_size, y_size), tf.uint8)
        green_tensor = tf.cast(green_tree.get_tensor(x_size, y_size), tf.uint8)
        blue_tensor = tf.cast(blue_tree.get_tensor(x_size, y_size), tf.uint8)
        #alpha_tensor = tf.cast(alpha_tree.get_tensor(x_size, y_size), tf.uint8)
        red_result = sess.run(red_tensor)
        green_result = sess.run(green_tensor)
        blue_result = sess.run(blue_tensor)
        #alpha_result = sess.run(alpha_tensor)
        auxiliary_array = [red_result, green_result, blue_result]
        result = np.stack(auxiliary_array, axis=2)
        fitness = (np.square(image_to_fit - result)).mean(axis=None)
        if best_fit != None and fitness < best_fit:
            print("Image saved!")
            matplotlib.pyplot.imsave('generated_images/' + str(current_generation) + '_' + str(current_individual) + '_generated_image.png', result)
        return fitness

if __name__ == "__main__":
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        with sess.graph.device("/gpu:1"):
                a = tf.constant(1)
                b = tf.constant(2)
                c = tf.math.maximum(a, b)
                alpha_result = sess.run(c)
                print(alpha_result)
