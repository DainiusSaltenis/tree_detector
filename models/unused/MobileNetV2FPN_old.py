from tensorflow.keras.layers import Input, Conv2DTranspose, BatchNormalization, ReLU, Conv2D, Lambda, MaxPooling2D, \
    Dropout, Add
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import constant, zeros
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import tensorflow as tf

from models.unused.MobileNetV2_old import MobileNetv2
from utils.decode import decode
from utils.losses import loss


def MobileNetV2_V1(num_classes=1, input_size=512, max_objects=500, score_threshold=0.1):
    output_size = input_size // 4
    image_input = Input(shape=(input_size, input_size, 3))
    hm_input = Input(shape=(output_size, output_size, num_classes))
    r_input = Input(shape=(max_objects, 1))
    reg_input = Input(shape=(max_objects, 2))
    reg_mask_input = Input(shape=(max_objects,))
    index_input = Input(shape=(max_objects,))

    backbone = MobileNetv2(image_input, num_classes, 1.0)

    bridge_1 = backbone.get_layer('activation_7').output
    bridge_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(bridge_1)
    bridge_1 = BatchNormalization()(bridge_1)
    bridge_1 = ReLU()(bridge_1)
    bridge_2 = backbone.get_layer('activation_13').output
    bridge_2 = Conv2D(128, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(bridge_2)
    bridge_2 = BatchNormalization()(bridge_2)
    bridge_2 = ReLU()(bridge_2)
    bridge_3 = backbone.get_layer('activation_27').output
    bridge_3 = Conv2D(256, (1, 1), strides=(1, 1), padding='same', kernel_initializer='he_normal')(bridge_3)
    bridge_3 = BatchNormalization()(bridge_3)
    bridge_3 = ReLU()(bridge_3)
    bridge_4 = backbone.get_layer('activation_34').output

    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(bridge_4)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, bridge_3])

    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, bridge_2])

    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Add()([x, bridge_1])

    x = Conv2D(64, kernel_size=(3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    hm = Conv2D(num_classes, 1, kernel_initializer='he_normal', activation='sigmoid')(x)
    r = Conv2D(1, 1, kernel_initializer='he_normal')(x)
    off = Conv2D(2, 1, kernel_initializer='he_normal')(x)

    loss_ = Lambda(loss, name='centernet_loss')(
        [hm, r, off, hm_input, r_input, reg_input, reg_mask_input, index_input])
    model = Model(inputs=[image_input, hm_input, r_input, reg_input, reg_mask_input, index_input], outputs=[loss_])

    # detections = decode(y1, y2, y3)
    detections = Lambda(lambda x: decode(*x, max_objects=max_objects))([hm, r, off])
    prediction_model = Model(inputs=image_input, outputs=detections)
    debug_model = Model(inputs=image_input, outputs=[hm, r, off])
    models = {'train_model': model, 'prediction_model': prediction_model, 'debug_model': debug_model}

    import os
    from tensorflow.keras.utils import plot_model
    os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\graphviz-2.38\\release\\bin'
    plot_model(model, to_file='mobilenetV2_V1.png', show_shapes=True, show_layer_names=True)

    prediction_model.summary()
    return models
