import cv2
import time
import os
import numpy as np


def draw_inputs_before_compute(image_batch, annotations_batch, mode='circle', verbose=True):
    folder_name = "outputs\\images_before_computation"
    for i, image in enumerate(image_batch):
        if mode == 'circle':
            __draw_image_before_compute_circle(image_batch[i], annotations_batch[i], folder_name, verbose)
        else:
            __draw_image_before_compute_bbox(image_batch[i], annotations_batch[i], folder_name, verbose)


def draw_inputs_computed(image_batch, heatmap_batch):
    folder_name = "outputs\\images_after_computation"
    for i, image in enumerate(image_batch):
        __draw_image_computed(image_batch[i, ...], heatmap_batch[i, ...], folder_name)
        #__draw_image_computed(image_batch, heatmap_batch, folder_name)


def draw_input_computed(image, heatmap):
    folder_name = "outputs\\images_after_computation"
    __draw_image_computed(image, heatmap, folder_name)


def draw_inputs_testing(image, predictions):
    folder_name = "outputs\\testing_predictions_final"
    __draw_image_test(image, predictions, folder_name)


def __draw_image_before_compute_bbox(image, annotations, folder_name, verbose=True):
    for a in annotations['bboxes']:
        cv2.rectangle(image, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 0, 255), 1)

    __save_to_file(image, folder_name, "before_computation_")


def __draw_image_before_compute_circle(image, annotations, folder_name, verbose=True):
    for a in annotations['bboxes']:
        cv2.circle(image, (int(a[0] + (a[2] - a[0]) / 2), int(a[1] + (a[3] - a[1]) / 2)), int((a[2] - a[0]) / 2),
                   (0, 0, 255), 2)

    __save_to_file(image, folder_name, "before_computation_")


def __draw_image_test(image, annotations, folder_name, verbose=True):
    for a in annotations:
        cv2.circle(image, (int(a[0]), int(a[1])), int(a[2]), (0, 0, 255), 3)

    __save_to_file(image, folder_name, "result_")


def __draw_image_computed(image, heatmap, folder_name, verbose=True):
    __save_to_file(image, folder_name, "original_computation_")

    w_r, h_r = int(image.shape[0] / heatmap.shape[0]), int(image.shape[1] / heatmap.shape[1])
    heatmap = cv2.resize(heatmap*255, (heatmap.shape[0] * w_r, heatmap.shape[1] * h_r))

    __save_to_file(heatmap, folder_name, "heatmap_")


def __save_to_file(image, folder_name, name, verbose=True, extension=".jpg"):
    name = str(time.time() * 1000) + name + extension
    full_name = os.path.join(folder_name, name)
    cv2.imwrite(full_name, image)

    if verbose:
        print("Image written: " + full_name)
