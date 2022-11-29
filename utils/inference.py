import cv2
import numpy as np
import math

from generators.utils import normalize_image
from utils.compute_overlap import compute_overlap, bb_intersection_over_union
from utils.convert import circle_to_bbox


def compute_tp_fp_fn(model_predictions, gt_predictions, upper_thr=9999, lower_thr=0, overlap_threshold=0.5):
    true_positives = 0
    model_predictions_boxes = [circle_to_bbox(x) for x in model_predictions]
    gt_predictions_boxes = [circle_to_bbox(x) for x in gt_predictions]
    i = 0
    while len(model_predictions_boxes) > i:
        model_prediction = model_predictions_boxes[i]
        for j, gt_prediction in enumerate(gt_predictions_boxes):
            overlap = bb_intersection_over_union(model_prediction, gt_prediction)
            if overlap >= overlap_threshold:
                true_positives += 1
                del model_predictions_boxes[i]
                del gt_predictions_boxes[j]
                i -= 1
                break
        i += 1
    return true_positives, len(model_predictions_boxes), len(gt_predictions_boxes)


def get_image(path, norm=True):
    image = read_image(path)
    #image[..., 0] -= 103.939
    #image[..., 1] -= 116.779
    #image[..., 2] -= 123.68
    if norm:
        image = normalize_image(image)
    return image


def resize_to_divisible(image, divisibility):
    h_div, w_div = image.shape[0] // divisibility, image.shape[1] // divisibility
    new_h, new_w = int(h_div * divisibility), int(w_div * divisibility)
    scale_h, scale_w = image.shape[0] / (h_div * divisibility), image.shape[1] / (w_div * divisibility)
    image = cv2.resize(image, (new_h, new_w), cv2.INTER_AREA)
    return image, scale_h, scale_w


def read_image(path):
    image = cv2.imread(path)
    image = image.astype(np.float32)
    return image


def slice_image(image, slice_size_h, slice_size_w, overlap):
    if image.shape[0] <= slice_size_h or image.shape[1] <= slice_size_w:
        return [[image]]
    h, w = image.shape[0], image.shape[1]
    slice_size_h_o, slice_size_w_o = slice_size_h - overlap, slice_size_w - overlap
    h_s, w_s = math.ceil((h - overlap) / slice_size_h_o), math.ceil((w - overlap) / slice_size_w_o)
    image_slices = []

    for i in range(h_s):
        row = []
        if i == h_s - 1:
            y1 = h - slice_size_h
            y2 = h
        else:
            y1 = i * slice_size_h_o
            y2 = i * slice_size_h_o + slice_size_h
        for j in range(w_s):
            if j == w_s - 1:
                x1 = w - slice_size_w
                x2 = w
            else:
                x1 = j * slice_size_w_o
                x2 = j * slice_size_w_o + slice_size_w

            image_slice = image[int(y1):int(y2), int(x1):int(x2), :]
            row.append(image_slice)

        image_slices.append(row)
    return image_slices


def transform_prediction_to_original_scale(prediction, threshold=0.2):
    return [[x[0] * 4, x[1] * 4, x[2] * 4, x[3]] for x in prediction if x[3] > threshold]


# Fuse all predictions from sliced images into one image
def fuse_slices(slice_predictions, slice_size_h, slice_size_w, overlap, image_h, image_w, prediction_threshold):
    predictions = []
    slice_size_h_o, slice_size_w_o = slice_size_h - overlap, slice_size_w - overlap
    for i, col in enumerate(slice_predictions):
        for j, image_slice in enumerate(col):
            current_offset_h, current_offset_w = i * slice_size_h_o, j * slice_size_w_o
            if i == len(slice_predictions) - 1:
                current_offset_h = image_h - slice_size_h
            if j == len(col) - 1:
                current_offset_w = image_w - slice_size_w
            for k, prediction in enumerate(image_slice):
                if prediction[3] > prediction_threshold:
                    predictions.append([prediction[0] * 4 + current_offset_w, prediction[1] * 4 + current_offset_h,
                                        prediction[2] * 4, prediction[3]])
    return predictions


# Kind of non-maximum suppression algorithm interpretation for circles
def nms_circles(predictions, radius_threshold=0.5):
    if len(predictions) == 0:
        return predictions
    for i in range(len(predictions)):
        for j in range(len(predictions)):
            if predictions[i] is None or len(predictions) == 0:
                break
            if predictions[j] is None:
                continue
            if i == j:
                continue
            p = predictions[i]
            o = predictions[j]
            x_dis = abs(p[0] - o[0])
            y_dis = abs(p[1] - o[1])
            dis = pow(pow(x_dis, 2) + pow(y_dis, 2), 0.5)
            radius = p[2] + o[2]
            overlap = radius - dis
            if overlap > p[2] * radius_threshold or overlap > o[2] * radius_threshold:
                if p[3] < o[3] and p in predictions:
                    # predictions.remove(p)
                    predictions[i] = None
                    break
                elif o in predictions:
                    # predictions.remove(o)
                    predictions[j] = None
    predictions = [x for x in predictions if x is not None]
    return predictions
