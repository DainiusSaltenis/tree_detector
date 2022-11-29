import tensorflow as tf
import os
from tensorflow.keras.utils import plot_model

from utils.losses import loss
from utils.decode import decode


class FPNModel:
    def __init__(self, input_size, max_objects=500, nms_score_threshold=0.1, plot=False, training=False, name='Model'):
        self.input_size = input_size if training else None
        self.max_objects = max_objects
        self.nms_score_threshold = nms_score_threshold
        self.training = training
        self.bridge_connection_names = self.initilialize_connection_names()
        self.output_size = input_size // 4 if input_size is not None else None
        # self.bridges_kernels = [128, 64, 32]
        # self.fpn_kernels = [128, 64, 32]
        self.plot = plot
        self.name = name
        self.models = self.construct_model()
        self.plot_model(self.models['train_model'])
        self.models['prediction_model'].summary()

    def construct_model(self):
        inputs = self.initialize_inputs()
        backbone = self.get_backbone(inputs['image_input'])
        backbone.summary()
        print("layers: " + str(len(backbone.layers)))
        bridges = self.get_bridges(backbone)
        last_layer = self.transition_block(backbone, self.bridge_connection_names[0])
        output = self.connect_fpn(bridges, last_layer)
        heads = self.connect_heads(output)
        return self.initialize_models(heads, inputs)

    def plot_model(self, train_model):
        if self.plot and self.training:
            os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\graphviz-2.38\\release\\bin'
            filename = self.name + "_FPN.png"
            plot_model(train_model, to_file=filename, show_shapes=True, show_layer_names=True)

    def initilialize_connection_names(self):
        return [None]

    def initialize_models(self, heads, inputs):
        models = {}
        models['train_model'] = self.train_model(heads, inputs)
        models['prediction_model'] = self.prediction_model(heads, inputs)
        models['debug_model'] = self.debug_model(heads, inputs)
        return models

    def debug_model(self, heads, inputs):
        model = tf.keras.Model(inputs=inputs['image_input'], outputs=[heads['hm'], heads['r'], heads['off']])
        return model

    def prediction_model(self, heads, inputs):
        detections = tf.keras.layers.Lambda(lambda x: decode(*x, max_objects=self.max_objects))\
            ([heads['hm'], heads['r'], heads['off']])
        model = tf.keras.Model(inputs=inputs['image_input'], outputs=detections)
        return model

    def train_model(self, heads, inputs):
        loss_ = tf.keras.layers.Lambda(loss, name='centernet_loss')(
            [heads['hm'], heads['r'], heads['off'], inputs['hm_input'], inputs['r_input'], inputs['reg_input'],
             inputs['reg_mask_input'], inputs['index_input']])
        model = tf.keras.Model(inputs=[inputs['image_input'], inputs['hm_input'], inputs['r_input'], inputs['reg_input'],
             inputs['reg_mask_input'], inputs['index_input']], outputs=[loss_])
        return model

    def connect_heads(self, output):
        heads = {}
        heads['hm'] = tf.keras.layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid')(output)
        heads['r'] = tf.keras.layers.Conv2D(1, 1, kernel_initializer='he_normal')(output)
        heads['off'] = tf.keras.layers.Conv2D(2, 1, kernel_initializer='he_normal')(output)
        return heads

    def connect_fpn(self, bridges, last_layer):
        for bridge, kernels in zip(bridges, self.fpn_kernels):
            last_layer = self.fpn_block(bridge, last_layer, kernels)
        return last_layer

    def fpn_block(self, bridge, last_layer, kernels):
        return None

    def transition_block(self, backbone, last_bridge_name):
        return backbone.get_layer(last_bridge_name).output

    def get_bridges(self, backbone):
        bridges = []
        for bridge, kernels in zip(self.bridge_connection_names[1::], self.bridges_kernels):
            bridges.append(self.bridge_connection(bridge, backbone, kernels))
        return bridges

    def bridge_connection(self, layer_name, backbone, kernels):
        return backbone.get_layer(layer_name).output

    def get_backbone(self, image_input) -> tf.keras.Model:
        pass

    def initialize_inputs(self):
        inputs = {}
        inputs['image_input'] = tf.keras.layers.Input(shape=(None, None, 3))
        inputs['hm_input'] = tf.keras.layers.Input(shape=(self.output_size, self.output_size, 1))
        inputs['r_input'] = tf.keras.layers.Input(shape=(self.max_objects, 1))
        inputs['reg_input'] = tf.keras.layers.Input(shape=(self.max_objects, 2))
        inputs['reg_mask_input'] = tf.keras.layers.Input(shape=(self.max_objects,))
        inputs['index_input'] = tf.keras.layers.Input(shape=(self.max_objects,))
        return inputs

    def get_model(self):
        return self.models