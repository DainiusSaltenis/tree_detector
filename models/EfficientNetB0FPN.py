from models.EfficientNet import EfficientNet
from models.FPNModel import FPNModel
import tensorflow as tf


class EfficientNetB0FPN(FPNModel):
    def __init__(self, input_size, max_objects=500, nms_score_threshold=0.1, plot=False, training=False,
                 name='EfficientNet_B0'):
        self.bridges_kernels = [128, 64, 32]
        self.fpn_kernels = [128, 64, 32]
        super().__init__(input_size, max_objects=max_objects, nms_score_threshold=nms_score_threshold, plot=plot,
                         training=training, name=name)

    def fpn_block(self, bridge, last_layer, kernels):
        x = tf.keras.layers.Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same',
                                            kernel_initializer='he_normal')(last_layer)
        x = tf.keras.layers.BatchNormalization()(x, training=self.training)
        x = tf.keras.layers.Activation(activation='relu')(x)
        x = tf.keras.layers.Add()([x, bridge])
        return x

    def transition_block(self, backbone, last_bridge_name):
        kernel_size = self.bridges_kernels[0]
        x = backbone.get_layer(last_bridge_name).output
        x = tf.keras.layers.Conv2D(kernel_size, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')\
            (x)
        x = tf.keras.layers.BatchNormalization()(x, training=self.training)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    def initilialize_connection_names(self):
        return ['block7a_se_excite', 'block6a_expand_activation', 'block4a_expand_activation',
                'block3a_expand_activation']

    def bridge_connection(self, layer_name, backbone, kernels):
        x = backbone.get_layer(layer_name).output
        x = tf.keras.layers.Conv2D(kernels, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x, training=self.training)
        x = tf.keras.layers.Activation(activation='relu')(x)
        return x

    def get_backbone(self, image_input) -> tf.keras.Model:
        model = EfficientNet(image_input, training=self.training, plot=self.plot)
        return model.get_model()