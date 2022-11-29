import json
import os
from datetime import datetime

import numpy as np

from utils import image_drawing
from utils.image_drawing import draw_inputs_testing
from utils.inference import get_image, resize_to_divisible, nms_circles, slice_image, fuse_slices, compute_tp_fp_fn
from utils.mainutils import print_times


def test_ordinary(models, args=None):
    divisibility = 32
    pred_threshold = 0.075

    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    model = models['prediction_model']

    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)

    times = []
    for image in dataset:
        image_path = os.path.join(args.dataset_path, image['filename'])
        image = get_image(image_path)
        image, scale_h, scale_w = resize_to_divisible(image, divisibility)
        image_vis = get_image(image_path, norm=False)
        image, _, _ = resize_to_divisible(image_vis, divisibility)

        model_input = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

        start = datetime.now()
        predictions = model.predict_on_batch(model_input)
        finish = datetime.now()
        time_diff = finish - start
        times.append(time_diff)

        predictions = [x for x in predictions if x[4] > pred_threshold]
        predictions = [[x[0] * scale_w, x[1] * scale_h, x[2] * scale_w, x[3] * scale_h, x[4]] for x in predictions]

        predictions = nms_circles(predictions, radius_threshold=1.5)
        draw_inputs_testing(image_vis, predictions)
    print_times(times)


def test_slicing(models, args=None):
    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    model = models['prediction_model']

    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)

    prediction_threshold = 0.1
    times = []
    for image in dataset:
        image_path = os.path.join(args.dataset_path, image['filename'])
        image = get_image(image_path)
        image_vis = get_image(image_path, norm=False)
        h, w = image.shape[0], image.shape[1]
        slices = slice_image(image, args.input_size, args.input_size, args.input_size / 4)

        start = datetime.now()
        predictions_on_slices = [[
            model.predict_on_batch(np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2])))[0]
            for image in col] for col in slices]
        finish = datetime.now()
        time_diff = finish - start
        times.append(time_diff)

        predictions = fuse_slices(predictions_on_slices, args.input_size, args.input_size,
                                  args.input_size / 4, h, w, prediction_threshold)
        predictions = nms_circles(predictions, radius_threshold=1.25)
        draw_inputs_testing(image_vis, predictions)
    print_times(times)


def test_on_dataset(models, args=None):
    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    divisibility = 32
    pred_threshold = 0.075

    dataset_path = os.path.join(args.dataset_path, args.test_json)

    with open(dataset_path) as f:
        dataset = json.load(f)

    analyse_dataset(dataset)

    model = models['prediction_model']

    print('Loading model, this may take a second...')
    model.load_weights(args.snapshot, by_name=True)

    times = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for dataset_image in dataset:
        image_path = os.path.join(args.dataset_path, dataset_image['filename'])
        image = get_image(image_path, norm=True)
        image, scale_h, scale_w = resize_to_divisible(image, divisibility)
        image_vis = get_image(image_path, norm=False)

        model_input = np.reshape(image, (1, image.shape[0], image.shape[1], image.shape[2]))

        start = datetime.now()
        predictions = model.predict_on_batch(model_input)
        finish = datetime.now()
        time_diff = finish - start
        times.append(time_diff.total_seconds())

        predictions_thr = []
        for x in predictions[0, ...]:
            if x[3] > pred_threshold:
                predictions_thr.append(x)
        predictions = predictions_thr
        #predictions = [x for x in predictions[0, ...] if x[3] > pred_threshold]
        predictions = [[x[0] * scale_h * 4, x[1] * scale_w * 4, x[2] * 4, x[3]] for x in predictions]

        predictions = nms_circles(predictions, radius_threshold=1.25)
        #draw_inputs_testing(image_vis, predictions)
        gt_predictions = [x['circle'] for x in dataset_image['objs']]
        tp, fp, fn = compute_tp_fp_fn(predictions, gt_predictions)
        true_positives += tp
        false_positives += fp
        false_negatives += fn
    print("Precision: " + str(true_positives / (true_positives + false_positives)))
    print("Recall: " + str(true_positives / (true_positives + false_negatives)))
    print_times(times[1::])


def analyse_dataset(dataset):
    images = 0
    trees = 0
    for image in dataset:
        images += 1
        trees += len(image['objs'])
    print("Trees: " + str(trees))
    print("Images: " + str(images))
