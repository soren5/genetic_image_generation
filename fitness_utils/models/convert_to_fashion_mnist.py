from keras.models import Sequential, load_model
from keras.datasets import fashion_mnist
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D

def convert_to_fashion_mnist(model_name):
    model = load_model(model_name)
    layer = model.layers[1]
    new_model = Sequential([
        Conv2D(
            filters=layer.filters,
            kernel_size=layer.kernel_size,
            strides=layer.strides,
            padding=layer.padding,
            data_format=layer.data_format,
            dilation_rate=layer.dilation_rate,
            activation=layer.activation,
            use_bias=layer.use_bias,
            kernel_initializer=layer.kernel_initializer,
            bias_initializer=layer.bias_initializer,
            kernel_regularizer=layer.kernel_regularizer,
            bias_regularizer=layer.bias_regularizer,
            activity_regularizer=layer.activity_regularizer,
            kernel_constraint=layer.kernel_constraint,
            bias_constraint=layer.bias_constraint,
            input_shape=(28, 28, 1),
        )
    ])
    print(model.summary())
    for i in range(2, len(model.layers)):
        layer = model.layers[i]
        if i != 0:
            if type(layer) == BatchNormalization:
                new_model.add(
                    BatchNormalization(
                        axis=layer.axis,
                        momentum=layer.momentum,
                        epsilon=layer.epsilon,
                        center=layer.center,
                        scale=layer.scale,
                        beta_initializer=layer.beta_initializer,
                        gamma_initializer=layer.gamma_initializer,
                        moving_mean_initializer=layer.moving_mean_initializer,
                        moving_variance_initializer=layer.moving_variance_initializer,
                        beta_regularizer=layer.beta_regularizer,
                        gamma_regularizer=layer.gamma_regularizer,
                        beta_constraint=layer.beta_constraint,
                        gamma_constraint=layer.gamma_constraint,
                    )
                )
            elif type(layer) == Conv2D:
                new_model.add(
                    Conv2D(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        data_format=layer.data_format,
                        dilation_rate=layer.dilation_rate,
                        activation=layer.activation,
                        use_bias=layer.use_bias,
                        kernel_initializer=layer.kernel_initializer,
                        bias_initializer=layer.bias_initializer,
                        kernel_regularizer=layer.kernel_regularizer,
                        bias_regularizer=layer.bias_regularizer,
                        activity_regularizer=layer.activity_regularizer,
                        kernel_constraint=layer.kernel_constraint,
                        bias_constraint=layer.bias_constraint
                    )
                )
            elif type(layer) == Flatten:
                new_model.add(
                    Flatten(
                        data_format=layer.data_format
                    )
                )
            elif type(layer) == Dense:
                new_model.add(
                    Dense(
                        units=layer.units,
                        activation=layer.activation,
                        use_bias=layer.use_bias,
                        kernel_initializer=layer.kernel_initializer,
                        bias_initializer=layer.bias_initializer,
                        kernel_regularizer=layer.kernel_regularizer,
                        bias_regularizer=layer.bias_regularizer,
                        activity_regularizer=layer.activity_regularizer,
                        kernel_constraint=layer.kernel_constraint,
                        bias_constraint=layer.bias_constraint
                    )
                )
            elif type(layer) == Dropout:
                new_model.add(
                    Dropout(
                        rate=layer.rate,
                        noise_shape=layer.noise_shape,
                        seed=layer.seed,
                    )
                )
    print(new_model.summary())
    new_model.save('model_7_0_fashionmnist.h5')
