import tensorflow.keras.backend as K
import sys

from main.Run.debug import debug_slicing, debug_ordinary
from main.FlaskApplication.deploy import deploy
from main.Run.export import export_to_onnx
from main.Run.test import test_ordinary, test_slicing, test_on_dataset
from main.Run.train import train
from models.CustomBackboneFPN import CustomBackboneFPN
from models.EfficientNetB0FPN import EfficientNetB0FPN
from models.MobileNetV2FPN import MobileNetV2FPN
from models.MobileNetV3FPN import MobileNetV3FPN
from models.ResNet50FPN import ResNet50FPN
from models.unused.MobileNetV2FPN_old import MobileNetV2_V1
from utils.mainutils import parse_args, initialize_tf_keras_session

loaded_models = {
    'MobileNetV2FPN': MobileNetV2FPN,
    'EfficientNetB0FPN': EfficientNetB0FPN,
    'MobileNetV3FPN': MobileNetV3FPN,
    'CustomNetFPN': CustomBackboneFPN,
    'ResNet50FPN': ResNet50FPN
}


run_modes = {
    "Train": (train, "Training model."),
    "Test1": (test_ordinary, "Testing model on standard images."),
    "Test2": (test_slicing, "Testing model on full scale images with cropping and fusing and common non-maximum "
                            "suppression."),
    "Test3": (test_on_dataset, "Testing model on standard images and computing precision and recall"),
    "Debug1": (debug_ordinary, "Debugging and writing heatmaps."),
    "Debug2": (debug_slicing, "Debugging with slicing and writing predictions with heatmaps."),
    "Export": (export_to_onnx, "Exporting model to onnx format."),
    "Run": (deploy, "Running the model in communication mode.")
}


def init_model(args=None):
    is_training = True if args.mode == 'Train' else False

    loaded_model = loaded_models[args.model_name](input_size=args.input_size, max_objects=1000, plot=True,
                                                  training=is_training, name=args.model_name)

    initialized_models = loaded_model.get_model()
    return initialized_models
    #return MobileNetV2_V1(input_size=args.input_size)


def run(models, args=None):
    print(run_modes[args.mode][1])
    if args.mode != 'Run':
        run_modes[args.mode][0](models, args)
    else:
        run_modes[args.mode][0](loaded_models, args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    K.set_session(initialize_tf_keras_session())

    models = init_model(args)
    run(models, args)


if __name__ == '__main__':
    main()
