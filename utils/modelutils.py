import tensorflow as tf


def relu6(x):
    kwargs = {'max_value': 6.0}
    return tf.keras.layers.Activation('relu', **kwargs)(x)


def h_swish_6(x):
    return x * tf.keras.activations.relu(x + 3, max_value=6) / 6


def h_swish(x):
    return x * tf.keras.activations.sigmoid(x)