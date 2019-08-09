import io
import numpy as np
from numpy import float64
import math
import matplotlib  
matplotlib.use('TkAgg')   
import matplotlib.pyplot as plt  
import timeit
import tensorflow as tf

dimensions = [[64, 64], [128, 128], [256, 256], [720, 480], [1280, 720], [1920, 1080]]
k = 0
for i in dimensions:
    k += 1
    x_dimension = i[0]
    y_dimension = i[1]
    for trial_count in range(30):
        image_array = np.zeros([x_dimension, y_dimension, 4], dtype=float)
        start_time = timeit.default_timer()
        for i in range(0, x_dimension):
            for j in range(0, y_dimension):
                image_array[i][j][0] = math.cos(i + j)
                image_array[i][j][1] = math.sin(i + j)
                image_array[i][j][2] = math.tan(i + j)
                image_array[i][j][3] = math.cos(i + j)
        elapsed = timeit.default_timer() - start_time
        matplotlib.pyplot.imsave(str(k) + '_' + str(trial_count) + '_traditional.png', image_array)
        print(str(k) + ',' + str(trial_count) + ',traditional,' + str(elapsed))
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        start_time = timeit.default_timer()
        with tf.Session(config=config) as sess:
            with sess.graph.device("/gpu:0"):
                x_values = np.linspace(0, x_dimension, x_dimension) 
                y_values = np.linspace(0, y_dimension, y_dimension)
                a = tf.constant(x_values, dtype=float64) 
                b = tf.constant(y_values, dtype=float64) 
                a = tf.expand_dims(a, 1)
                b = tf.expand_dims(b, 0)
                
                red = tf.cos(tf.add(a, b))
                green = tf.sin(tf.add(a, b))
                blue = tf.tan(tf.add(a, b))
                alpha = tf.cos(tf.add(a, b))

                red_result = sess.run(red)
                green_result = sess.run(green)
                blue_result = sess.run(blue)
                alpha_result = sess.run(alpha)
                
                elapsed = timeit.default_timer() - start_time
                result = np.stack((red_result, green_result, blue_result, alpha_result), axis=2)
                #matplotlib.pyplot.imsave('generated_images/' + str(k) + '_' + str(trial_count) + '_tensorflow.png', result)
                print(str(k) + ',' + str(trial_count) + ',tensorflow,' + str(elapsed))
                

