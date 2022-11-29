from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import string
import collections

from six.moves import xrange
import tensorflow as tf
import tensorflow.keras.layers as layers
from models.Backbone import Backbone


class EfficientNet(Backbone):
    def __init__(self, input_layer, training=False, plot=False, name='EfficientNet', width_coefficient=1.0,
                 depth_coefficient=1.0, default_resolution=1024, depth_divisor=8):
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_resolution = default_resolution
        self.depth_divisor = depth_divisor

        super().__init__(input_layer, training, plot, name)

    def init_model(self) -> tf.keras.Model:
        BlockArgs = collections.namedtuple('BlockArgs', ['kernel_size', 'num_repeat', 'input_filters', 'output_filters',
                                                         'expand_ratio', 'id_skip', 'strides', 'se_ratio'])

        BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

        DEFAULT_BLOCKS_ARGS = [
            BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
                      expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
                      expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
            BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
                      expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
            BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
                      expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
        ]

        CONV_KERNEL_INITIALIZER = {
            'class_name': 'VarianceScaling',
            'config': {
                'scale': 2.0,
                'mode': 'fan_out',
                'distribution': 'normal'
            }
        }

        def get_swish(x):
            return x * tf.keras.activations.sigmoid(x)

        def round_filters(filters, width_coefficient, depth_divisor):
            """Round number of filters based on width multiplier."""

            filters *= width_coefficient
            new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
            new_filters = max(depth_divisor, new_filters)
            if new_filters < 0.9 * filters:
                new_filters += depth_divisor
            return int(new_filters)

        def round_repeats(repeats, depth_coefficient):
            """Round number of repeats based on depth multiplier."""

            return int(math.ceil(depth_coefficient * repeats))

        def mb_conv_block(inputs, block_args, activation, prefix='', ):
            """Mobile Inverted Residual Bottleneck."""

            has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)

            filters = block_args.input_filters * block_args.expand_ratio
            if block_args.expand_ratio != 1:
                x = layers.Conv2D(filters, 1,
                                  padding='same',
                                  use_bias=False,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'expand_conv')(inputs)
                x = layers.BatchNormalization(name=prefix + 'expand_bn')(x)
                x = layers.Activation(activation, name=prefix + 'expand_activation')(x)
            else:
                x = inputs

            x = layers.DepthwiseConv2D(block_args.kernel_size,
                                       strides=block_args.strides,
                                       padding='same',
                                       use_bias=False,
                                       depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                       name=prefix + 'dwconv')(x)
            x = layers.BatchNormalization(name=prefix + 'bn')(x)
            x = layers.Activation(activation, name=prefix + 'activation')(x)

            if has_se:
                num_reduced_filters = max(1, int(
                    block_args.input_filters * block_args.se_ratio
                ))
                se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

                target_shape = (1, 1, filters)
                se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
                se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                          activation=activation,
                                          padding='same',
                                          use_bias=True,
                                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                                          name=prefix + 'se_reduce')(se_tensor)
                se_tensor = layers.Conv2D(filters, 1,
                                          activation='sigmoid',
                                          padding='same',
                                          use_bias=True,
                                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                                          name=prefix + 'se_expand')(se_tensor)
                x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

            x = layers.Conv2D(block_args.output_filters, 1,
                              padding='same',
                              use_bias=False,
                              kernel_initializer=CONV_KERNEL_INITIALIZER,
                              name=prefix + 'project_conv')(x)
            x = layers.BatchNormalization(name=prefix + 'project_bn')(x)
            if block_args.id_skip and all(
                    s == 1 for s in block_args.strides
            ) and block_args.input_filters == block_args.output_filters:
                x = layers.add([x, inputs], name=prefix + 'add')

            return x

        img_input = self.input
        blocks_args = DEFAULT_BLOCKS_ARGS

        activation = get_swish

        # Build stem
        x = img_input
        x = layers.Conv2D(round_filters(32, self.width_coefficient, self.depth_divisor), 3, strides=(2, 2),
                          padding='same', use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER,
                          name='stem_conv')(x)
        x = layers.BatchNormalization(name='stem_bn')(x)
        x = layers.Activation(activation, name='stem_activation')(x)

        block_num = 0
        for idx, block_args in enumerate(blocks_args):
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self.width_coefficient, self.depth_divisor),
                output_filters=round_filters(block_args.output_filters,
                                             self.width_coefficient, self.depth_divisor),
                num_repeat=round_repeats(block_args.num_repeat, self.depth_coefficient))

            x = mb_conv_block(x, block_args,
                              activation=activation,
                              prefix='block{}a_'.format(idx + 1))
            block_num += 1
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                for bidx in xrange(block_args.num_repeat - 1):
                    block_prefix = 'block{}{}_'.format(
                        idx + 1,
                        string.ascii_lowercase[bidx + 1]
                    )
                    x = mb_conv_block(x, block_args,
                                      activation=activation,
                                      prefix=block_prefix)
                    block_num += 1

        model = tf.keras.Model(img_input, x)

        return model
