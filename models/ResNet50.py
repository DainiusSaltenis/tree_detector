from models.Backbone import Backbone
from tensorflow.keras.applications.resnet50 import ResNet50 as RN50
import tensorflow as tf


class ResNet50(Backbone):
    def __init__(self, input_layer, training=False, plot=False, name='ResNet50'):
        self.kernel_initializer = 'he_normal'

        super().__init__(input_layer, training, plot, name)

    def init_model(self) -> tf.keras.Model:
        resnet = RN50(input_tensor=self.input, include_top=False)
        return resnet