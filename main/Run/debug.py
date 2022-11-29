import json
import os
from datetime import datetime

import numpy as np

from utils import image_drawing
from utils.inference import get_image, resize_to_divisible, slice_image, transform_prediction_to_original_scale
from utils.mainutils import print_times


def debug_ordinary(models, args=None):
    divisibility = 32

    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    model = models['prediction_model']
    debug_model = models['debug_model']

    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)

    times = []
    for image in dataset:
        image_path = os.path.join(args.dataset_path, image['filename'])
        image = get_image(image_path)
        image, _, _ = resize_to_divisible(image, divisibility)
        image_vis = get_image(image_path, norm=False)
        image_vis, _, _ = resize_to_divisible(image_vis, divisibility)
        model_input = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

        start = datetime.now()
        heatmap = debug_model.predict_on_batch(model_input)[0]
        finish = datetime.now()
        time_diff = finish - start
        times.append(time_diff)

        image_drawing.draw_inputs_computed(np.expand_dims(image_vis, axis=0), heatmap)
    print_times(times)


def debug_slicing(models, args=None):
    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    model = models['prediction_model']
    debug_model = models['debug_model']

    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)

    times = []
    for image in dataset:
        image_path = os.path.join(args.dataset_path, image['filename'])
        image = get_image(image_path)
        image_vis = get_image(image_path, norm=False)
        slices = slice_image(image, args.input_size, args.input_size, args.input_size / 4)
        slices_vis = slice_image(image_vis, args.input_size, args.input_size, args.input_size / 4)
        predictions_on_slices = [[
            model.predict_on_batch(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))[0]
            for image in col] for col in slices]

        start = datetime.now()
        heatmaps_on_slices = [[
            debug_model.predict_on_batch(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))[0]
            for image in col] for col in slices]
        finish = datetime.now()
        time_diff = finish - start
        times.append(time_diff)

        for i, row in enumerate(slices_vis):
            for j, slice in enumerate(row):
                image_vis = slice.copy()
                predictions_on_slices[i][j] = transform_prediction_to_original_scale(predictions_on_slices[i][j],
                                                                                     threshold=0.2)
                image_drawing.draw_input_computed(image_vis, heatmaps_on_slices[i][j][0, ...])
                image_drawing.draw_inputs_testing(image_vis, predictions_on_slices[i][j])

    print_times(times)
