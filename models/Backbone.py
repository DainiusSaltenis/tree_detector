import os
from typing import Tuple, Dict
import tensorflow as tf
import abc

from tensorflow.keras.utils import plot_model


class Backbone:
    def __init__(self, input_layer, training=False, plot=False, name='Model'):
        self.input = input_layer
        self.training = training
        self.plot = plot
        self.name = name
        self.model = self.init_model()
        self.plot_model(self.model)

    def init_model(self) -> tf.keras.Model:
        pass

    def get_model(self) -> tf.keras.Model:
        return self.model

    def plot_model(self, model):
        if self.plot and self.training:
            os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\graphviz-2.38\\release\\bin'
            filename = self.name + "_backbone.png"
            plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

