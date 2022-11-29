import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, DepthwiseConv2D, Activation, Input, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply

from models.Backbone import Backbone


class MobileNetV3(Backbone):
    def __init__(self, input_layer, training=False, plot=False, name='MobileNetV3_large'):
        self.config_list = self.get_config_list()
        super().__init__(input_layer, training, plot, name)

    def get_config_list(self):
        return [[16, (3, 3), (1, 1), 16, False, False, False, 'RE', 0],
                 [24, (3, 3), (2, 2), 64, False, False, False, 'RE', 1],
                 [24, (3, 3), (1, 1), 72, False, True, False, 'RE', 2],
                 [40, (5, 5), (2, 2), 72, False, False, True, 'RE', 3],
                 [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 4],
                 [40, (5, 5), (1, 1), 120, False, True, True, 'RE', 5],
                 [80, (3, 3), (2, 2), 240, False, False, False, 'HS', 6],
                 [80, (3, 3), (1, 1), 200, False, True, False, 'HS', 7],
                 [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 8],
                 [80, (3, 3), (1, 1), 184, False, True, False, 'HS', 9],
                 [112, (3, 3), (1, 1), 480, False, False, True, 'HS', 10],
                 [112, (3, 3), (1, 1), 672, False, True, True, 'HS', 11],
                 [160, (5, 5), (1, 1), 672, False, False, True, 'HS', 12],
                 [160, (5, 5), (2, 2), 672, False, True, True, 'HS', 13],
                 [160, (5, 5), (1, 1), 960, False, True, True, 'HS', 14]]

    def init_model(self) -> tf.keras.Model:
        def Hswish(x):
            return x * tf.nn.relu6(x + 3) / 6

        def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
            x = Conv2D(filters, kernel, strides=strides, padding=padding, use_bias=is_use_bias)(_inputs)
            x = BatchNormalization()(x)
            if activation == 'RE':
                x = ReLU(name=name)(x)
            elif activation == 'HS':
                x = Activation(Hswish, name=name)(x)
            else:
                raise NotImplementedError
            return x


        def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
            x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same')(_inputs)
            x = BatchNormalization()(x)
            if is_use_se:
                x = __se_block(x)
            if activation == 'RE':
                x = ReLU()(x)
            elif activation == 'HS':
                x = Activation(Hswish)(x)
            else:
                raise NotImplementedError
            return x


        def __global_depthwise_block(_inputs):
            assert _inputs._keras_shape[1] == _inputs._keras_shape[2]
            kernel_size = _inputs._keras_shape[1]
            x = DepthwiseConv2D((kernel_size, kernel_size), strides=(1, 1), depth_multiplier=1, padding='valid')(_inputs)
            return x


        def __se_block(_inputs, ratio=4, pooling_type='avg'):
            filters = _inputs.shape[3]
            se_shape = (1, 1, filters)
            if pooling_type == 'avg':
                se = GlobalAveragePooling2D()(_inputs)
            elif pooling_type == 'depthwise':
                se = __global_depthwise_block(_inputs)
            else:
                raise NotImplementedError
            se = Reshape(se_shape)(se)
            se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
            se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
            return multiply([_inputs, se])


        def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True,
                               is_use_se=True, activation='RE', num_layers=0, *args):
            with tf.name_scope('bottleneck_block'):
                bottleneck_dim = expansion_dim

                x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias,
                                   activation=activation)

                x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation,
                                      num_layers=num_layers)

                x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
                x = BatchNormalization()(x)

                if shortcut and strides == (1, 1):
                    in_dim = K.int_shape(_inputs)[-1]
                    if in_dim != out_dim:
                        ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
                        x = Add()([x, ins])
                    else:
                        x = Add()([x, _inputs])
            return x

        inputs = self.input

        # ** feature extraction layers
        net = __conv2d_block(inputs, 16, kernel=(3, 3), strides=(2, 2), is_use_bias=False, padding='same', activation='HS')

        for config in self.config_list:
            net = __bottleneck_block(net, *config)

        net = __conv2d_block(net, 960, kernel=(3, 3), strides=(1, 1), is_use_bias=True, padding='same', activation='HS',
                             name='output_map')

        model = Model(inputs=inputs, outputs=net)

        return model


