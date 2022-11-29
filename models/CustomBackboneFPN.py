from models.CustomBackbone import CustomBackbone
from models.FPNModel import FPNModel
import tensorflow as tf


class CustomBackboneFPN(FPNModel):
    def __init__(self, input_size, max_objects=500, nms_score_threshold=0.1, plot=False, training=False,
                 name='CustomNetFPN'):
        self.bridges_kernels = [96, 64, 48]
        self.fpn_kernels = [96, 64, 48]
        super().__init__(input_size, max_objects=max_objects, nms_score_threshold=nms_score_threshold, plot=plot,
                         training=training, name=name)

    def fpn_block(self, bridge, last_layer, kernels):
        x = tf.keras.layers.Conv2DTranspose(kernels, (2, 2), strides=(2, 2), padding='same',
                                            kernel_initializer='he_normal')(last_layer)
        x = tf.keras.layers.BatchNormalization()(x, training=self.training)
        x = tf.keras.layers.Activation(activation='relu')(x)
        #x = tf.keras.layers.Add()([x, bridge])
        x = tf.keras.layers.Concatenate()([x, bridge])
        x = tf.keras.layers.Conv2D(filters=kernels, kernel_size=(1, 1), padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x, training=self.training)
        x = tf.keras.layers.Activation(activation='relu')(x)

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
        return ['add_16', ['add_12', 'add_6'], 'add_4', 'add_1']

    def bridge_connection(self, layer_name, backbone, kernels):
        if isinstance(layer_name, str):
            x = backbone.get_layer(layer_name).output
            x = tf.keras.layers.Conv2D(kernels, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x, training=self.training)
            x = tf.keras.layers.Activation(activation='relu')(x)
        else:
            x1 = backbone.get_layer(layer_name[0]).output
            x1 = tf.keras.layers.Conv2D(kernels, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')\
                    (x1)
            x1 = tf.keras.layers.BatchNormalization()(x1, training=self.training)
            x1 = tf.keras.layers.Activation(activation='relu')(x1)

            x2 = backbone.get_layer(layer_name[1]).output
            x2 = tf.keras.layers.Conv2D(kernels, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')\
                    (x2)
            x2 = tf.keras.layers.BatchNormalization()(x2, training=self.training)
            x2 = tf.keras.layers.Activation(activation='relu')(x2)

            x = tf.keras.layers.Add()([x1, x2])

        return x

    def get_backbone(self, image_input) -> tf.keras.Model:
        model = CustomBackbone(image_input, training=self.training, plot=self.plot)
        return model.get_model()