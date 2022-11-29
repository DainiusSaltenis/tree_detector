import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import BatchNormalization, Add, DepthwiseConv2D
from tensorflow.keras import backend as K

from models.Backbone import Backbone


class MobileNetV2(Backbone):
    def __init__(self, input_layer, training=False, plot=False, name='Model', alpha=1.0):
        self.alpha = alpha
        super().__init__(input_layer, training, plot, name)

    def init_model(self) -> tf.keras.Model:
        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def _conv_block(inputs, filters, kernel, strides):
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

            x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
            x = BatchNormalization(axis=channel_axis)(x)
            return ReLU(max_value=6)(x)

        def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            tchannel = K.int_shape(inputs)[channel_axis] * t
            cchannel = int(filters * alpha)

            x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

            x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
            x = BatchNormalization(axis=channel_axis)(x)
            x = ReLU(max_value=6)(x)

            x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
            x = BatchNormalization(axis=channel_axis)(x)

            if r:
                x = Add()([x, inputs])

            return x

        def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
            x = _bottleneck(inputs, filters, kernel, t, alpha, strides)

            for i in range(1, n):
                x = _bottleneck(x, filters, kernel, t, alpha, 1, True)

            return x

        alpha = self.alpha
        first_filters = _make_divisible(32 * alpha, 8)
        x = _conv_block(self.input, first_filters, (3, 3), strides=(2, 2))

        x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
        x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
        x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
        x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
        x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
        x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
        x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)

        return Model(self.input, x)
