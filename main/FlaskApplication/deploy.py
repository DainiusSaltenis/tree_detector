import io
import numpy as np

import flask
from PIL import Image

import tensorflow as tf

from generators.utils import normalize_image
from utils.inference import resize_to_divisible, fuse_slices, slice_image, transform_prediction_to_original_scale


def deploy(models=None, args=None):
    app = flask.Flask(__name__)
    from utils.inference import nms_circles

    @app.route("/predict", methods=["POST"])
    def rest_predict():
        data = {"success": False}
        divisibility = 32
        prediction_threshold = 0.1
        size_limit = 1024

        # ensure an image was properly uploaded to our endpoint
        if flask.request.method == "POST":
            files = flask.request.files.keys()

            # Locally loading a model due to multithreading
            loaded_model = models[args.model_name](input_size=args.input_size, max_objects=1000, plot=True,
                                                   training=True, name=args.model_name)
            model = loaded_model.get_model()['prediction_model']

            print('Loading model, this may take a second...')
            model.load_weights(args.snapshot, by_name=True)

            for file in files:
                # read the image in PIL format
                image = flask.request.files[file].read()
                image = Image.open(io.BytesIO(image))

                image = tf.keras.preprocessing.image.img_to_array(image)
                image = image.astype(np.float32)
                image = normalize_image(image)

                if image.shape[0] < size_limit or image.shape[1] < size_limit:
                    image, scale_h, scale_w = resize_to_divisible(image, divisibility)

                    predictions = \
                        model.predict_on_batch(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))[0]

                    predictions = transform_prediction_to_original_scale(predictions, prediction_threshold)

                    predictions = nms_circles(predictions, radius_threshold=1.25)
                else:
                    h, w = image.shape[0], image.shape[1]
                    slices = slice_image(image, args.input_size, args.input_size, args.input_size / 4)
                    predictions_on_slices = [[
                        model.predict_on_batch(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))
                        [0] for image in col] for col in slices]

                    predictions = fuse_slices(predictions_on_slices, args.input_size, args.input_size,
                                              args.input_size / 4, h, w, prediction_threshold)
                    predictions = nms_circles(predictions, radius_threshold=1.25)

                predictions = [str(x) for x in predictions]
                data[file] = []

                for pred in predictions:
                    data[file].append(pred)
                data["success"] = True
        return flask.jsonify(data)

    app.run()