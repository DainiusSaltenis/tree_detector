from models.Backbone import Backbone
import tensorflow as tf


class CustomBackbone(Backbone):
    def __init__(self, input_layer, training=False, plot=False, name='CustomNet'):
        self.kernel_initializer = 'he_normal'

        self.model_config = [
            {"filters": 16, "expand_ratio": 3, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": False},
            {"filters": 32, "expand_ratio": 3.5, "kernel": (3, 3), "stride": (2, 2), "act": "relu", "ext": True},
            {"filters": 32, "expand_ratio": 3.5, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 56, "expand_ratio": 3.25, "kernel": (5, 5), "stride": (2, 2), "act": "relu", "ext": True},
            {"filters": 56, "expand_ratio": 3.25, "kernel": (5, 5), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 56, "expand_ratio": 3.25, "kernel": (5, 5), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 3.25, "kernel": (5, 5), "stride": (2, 2), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 3.25, "kernel": (5, 5), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 98, "expand_ratio": 4.25, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 170, "expand_ratio": 4.25, "kernel": (5, 5), "stride": (2, 2), "act": "relu", "ext": True},
            {"filters": 170, "expand_ratio": 4.25, "kernel": (5, 5), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 170, "expand_ratio": 6, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True},
            {"filters": 170, "expand_ratio": 6, "kernel": (3, 3), "stride": (1, 1), "act": "relu", "ext": True}
            #{"filters": 297, "expand_ratio": 3.5, "kernel": (3, 3), "stride": (2, 2), "act": "lrelu", "ext": True},
            #{"filters": 297, "expand_ratio": 3.5, "kernel": (3, 3), "stride": (1, 1), "act": "lrelu", "ext": True},
        ]

        super().__init__(input_layer, training, plot, name)

    def init_model(self) -> tf.keras.Model:
        def activation(x, name='relu'):
            if name == 'relu':
                return tf.keras.layers.Activation(activation='relu')(x)
            if name == 'lrelu':
                return tf.keras.layers.LeakyReLU()(x)
            raise ValueError('A chosen activation function was not specified.')

        def bottleneck_block(x, filters, expand_ratio, dw_conv_kernel, stride, activation_name, is_odd,
                             use_extension=True):
            expand_conv = tf.keras.layers.Conv2D(int(filters * expand_ratio), kernel_size=(1, 1), padding='same',
                                                 use_bias=False, kernel_initializer=self.kernel_initializer)(x)
            expand_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.997)(expand_conv, training=self.training)
            expand_conv_act = activation(expand_conv_bn, activation_name)
            dw_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=dw_conv_kernel, strides=stride, padding='same',
                                                      use_bias=False, kernel_initializer=self.kernel_initializer)\
                                                     (expand_conv_act)
            dw_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.997)(dw_conv, training=self.training)
            shrink_conv = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False,
                                                 kernel_initializer=self.kernel_initializer)(dw_conv_bn)
            shrink_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.997)(shrink_conv, training=self.training)
            shrink_conv_act = activation(shrink_conv_bn, activation_name)

            #if is_odd and use_extension:
            if use_extension:
                if stride == (1, 1):
                    #output = tf.keras.layers.Concatenate()([shrink_conv_act, x])
                    output = tf.keras.layers.Add()([shrink_conv_act, x])
                else:
                    dw_conv_cn = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=stride,
                                                                 padding='same', use_bias=False,
                                                                 kernel_initializer=self.kernel_initializer)\
                                                                 (x)
                    dw_conv_bn_cn = tf.keras.layers.BatchNormalization(momentum=0.997)\
                                                                      (dw_conv_cn, training=self.training)
                    shrink_conv_cn = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same', use_bias=False,
                                                            kernel_initializer=self.kernel_initializer)(dw_conv_bn_cn)
                    shrink_conv_bn_cn = tf.keras.layers.BatchNormalization(momentum=0.997) \
                        (shrink_conv_cn, training=self.training)
                    shrink_conv_cn_act = activation(shrink_conv_bn_cn, activation_name)
                    output = tf.keras.layers.Add()([shrink_conv_cn_act, shrink_conv_act])
                    #output = tf.keras.layers.Concatenate()([dw_conv_act_cn, shrink_conv_act])
            else:
                output = shrink_conv_act
            return output

        input_layer = self.input

        init_conv = tf.keras.layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False,
                                           kernel_initializer=self.kernel_initializer)(input_layer)
        expand_conv_bn = tf.keras.layers.BatchNormalization(momentum=0.997)(init_conv, training=self.training)
        expand_conv_act = activation(expand_conv_bn, name='lrelu')

        out = expand_conv_act

        for i, config in enumerate(self.model_config):
            out = bottleneck_block(out,
                                   filters=config['filters'],
                                   expand_ratio=config['expand_ratio'],
                                   dw_conv_kernel=config['kernel'],
                                   stride=config['stride'],
                                   activation_name=config['act'],
                                   is_odd=True if i % 2 == 1 else False,
                                   use_extension=config['ext'])

        model = tf.keras.Model(input_layer, out)
        return model



