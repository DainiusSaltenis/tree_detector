from unittest import TestCase

import cv2
import numpy as np

import utils
from utils.compute_overlap import bb_intersection_over_union
from utils.inference import get_image, resize_to_divisible, read_image, slice_image, fuse_slices, nms_circles, \
    compute_tp_fp_fn
from unittest import TestCase

from utils.convert import pascal_to_coco, coco_to_pascal, circle_to_bbox


class Test(TestCase):
    def test_pascal_to_coco_1(self):
        array_in = [[100, 100, 200, 150]]
        array_out = [[100, 100, 100, 50]]
        pascal_to_coco(array_in)
        self.assertEqual(array_in, array_out)

    def test_pascal_to_coco_2(self):
        array_in = [[0, 0, 200, 150]]
        array_out = [[0, 0, 200, 150]]
        pascal_to_coco(array_in)
        self.assertEqual(array_in, array_out)

    def test_pascal_to_coco_3(self):
        array_in = [[-10, -10, 10, 10]]
        array_out = [[-10, -10, 20, 20]]
        pascal_to_coco(array_in)
        self.assertEqual(array_in, array_out)


    def test_coco_to_pascal_1(self):
        array_in = [[100, 100, 100, 50]]
        array_out = [[100, 100, 200, 150]]
        coco_to_pascal(array_in)
        self.assertEqual(array_in, array_out)

    def test_coco_to_pascal_2(self):
        array_in = [[100, 100, 0, 0]]
        array_out = [[100, 100, 100, 100]]
        coco_to_pascal(array_in)
        self.assertEqual(array_in, array_out)

    def test_coco_to_pascal_3(self):
        array_in = [[-10, -10, 20, 20]]
        array_out = [[-10, -10, 10, 10]]
        coco_to_pascal(array_in)
        self.assertEqual(array_in, array_out)


    def test_circle_to_bbox_1(self):
        input = [10, 10, 5]
        output = [5, 5, 15, 15]
        circle = circle_to_bbox(input)
        self.assertEqual(output, circle)

    def test_circle_to_bbox_2(self):
        input = [10, 10, 15]
        output = [-5, -5, 25, 25]
        circle = circle_to_bbox(input)
        self.assertEqual(output, circle)

    def test_circle_to_bbox_3(self):
        input = [10, 5, 10]
        output = [0, -5, 20, 15]
        circle = circle_to_bbox(input)
        self.assertEqual(output, circle)

    def test_circle_to_bbox_4(self):
        input = [1000, 500, 10]
        output = [990, 490, 1010, 510]
        circle = circle_to_bbox(input)
        self.assertEqual(output, circle)


    def test_get_image_1(self):
        shape = (1015, 1021, 3)
        image = get_image('../test.jpg')
        self.assertEqual(image.shape, shape)

    def test_get_image_2(self):
        image = get_image('../test.jpg')
        if type(image) is not np.ndarray:
            self.fail()

    def test_get_image_3(self):
        image = get_image('../test.jpg')
        out_of_bounds= np.where(image > 1.000001)
        if out_of_bounds[0].shape[0] > 0 or out_of_bounds[1].shape[0] > 0 or out_of_bounds[2].shape[0] > 0:
            self.fail()

    def test_get_image_4(self):
        image = get_image('../test.jpg')
        out_of_bounds= np.where(image < -1.000001)
        if out_of_bounds[0].shape[0] > 0 or out_of_bounds[1].shape[0] > 0 or out_of_bounds[2].shape[0] > 0:
            self.fail()

    def test_get_image_5(self):
        image = get_image('../test.jpg', norm=False)
        out_of_bounds = np.where(image > 255.000001)
        if out_of_bounds[0].shape[0] > 0 or out_of_bounds[1].shape[0] > 0 or out_of_bounds[2].shape[0] > 0:
            self.fail()

    def test_get_image_6(self):
        image = get_image('../test.jpg', norm=False)
        out_of_bounds= np.where(image < -0.000001)
        if out_of_bounds[0].shape[0] > 0 or out_of_bounds[1].shape[0] > 0 or out_of_bounds[2].shape[0] > 0:
            self.fail()


    def test_resize_to_divisible_1(self):
        shape = (1000, 1000, 3)
        divisibility = 100
        image = get_image('../test.jpg')
        coef1, coef2 = image.shape[0] / 1000, image.shape[1] / 1000,
        image = resize_to_divisible(image, divisibility)
        self.assertEqual(image[0].shape, shape)
        self.assertEqual(image[1], coef1)
        self.assertEqual(image[2], coef2)

    def test_resize_to_divisible_2(self):
        shape = (500, 1000, 3)
        divisibility = 2
        image = get_image('../test.jpg')
        image = cv2.resize(image, (501, 1000))
        coef1, coef2 = image.shape[0] / 1000, image.shape[1] / 500,
        image = resize_to_divisible(image, divisibility)
        self.assertEqual(image[0].shape, shape)
        self.assertEqual(image[1], coef1)
        self.assertEqual(image[2], coef2)

    def test_resize_to_divisible_3(self):
        shape = (1500, 1000, 3)
        divisibility = 2
        image = get_image('../test.jpg')
        image = cv2.resize(image, (1501, 1000))
        coef1, coef2 = image.shape[0] / 1000, image.shape[1] / 1500,
        image = resize_to_divisible(image, divisibility)
        self.assertEqual(image[0].shape, shape)
        self.assertEqual(image[1], coef1)
        self.assertEqual(image[2], coef2)


    def test_read_image(self):
        shape = (1015, 1021, 3)
        image = utils.inference.read_image('../test.jpg')
        self.assertEqual(image.shape, shape)
        if type(image) is not np.ndarray:
            self.fail()


    def test_slice_image_1(self):
        slice_l = 128
        overlap = 64
        image = get_image('../test.jpg')
        slice_amount_h = int(image.shape[0] / (slice_l - overlap))
        slice_amount_w = int(image.shape[1] / (slice_l - overlap))
        slices = slice_image(image, slice_l, slice_l, overlap=overlap)
        self.assertEqual(len(slices), slice_amount_h)
        for slice in slices:
            self.assertEqual(len(slice), slice_amount_w)

    def test_slice_image_2(self):
        slice_l = 128
        overlap = 0
        image = get_image('../test.jpg')
        slice_amount_h = int(image.shape[0] / (slice_l - overlap)) + 1
        slice_amount_w = int(image.shape[1] / (slice_l - overlap)) + 1
        slices = slice_image(image, slice_l, slice_l, overlap=overlap)
        self.assertEqual(len(slices), slice_amount_h)
        for slice in slices:
            self.assertEqual(len(slice), slice_amount_w)

    def test_slice_image_3(self):
        slice_l = 128
        overlap = 1
        image = get_image('../test.jpg')
        slice_amount_h = int(image.shape[0] / (slice_l - overlap)) + 1
        slice_amount_w = int(image.shape[1] / (slice_l - overlap)) + 1
        slices = slice_image(image, slice_l, slice_l, overlap=overlap)
        self.assertEqual(len(slices), slice_amount_h)
        for slice in slices:
            self.assertEqual(len(slice), slice_amount_w)

    def test_slice_image_4(self):
        slice_l = 1024
        overlap = 0
        image = get_image('../test.jpg')
        slice_amount_h = int(image.shape[0] / (slice_l - overlap)) + 1
        slice_amount_w = int(image.shape[1] / (slice_l - overlap)) + 1
        slices = slice_image(image, slice_l, slice_l, overlap=overlap)
        self.assertEqual(len(slices), slice_amount_h)
        for slice in slices:
            self.assertEqual(len(slice), slice_amount_w)

    def test_slice_image_5(self):
        slice_l = 1
        overlap = 0
        image = get_image('../test.jpg')
        slice_amount_h = int(image.shape[0] / (slice_l - overlap))
        slice_amount_w = int(image.shape[1] / (slice_l - overlap))
        slices = slice_image(image, slice_l, slice_l, overlap=overlap)
        self.assertEqual(len(slices), slice_amount_h)
        for slice in slices:
            self.assertEqual(len(slice), slice_amount_w)


    def test_fuse_slices_1(self):
        slices_in = [
            [[np.array([2, 2, 2, 0.5])], [np.array([3, 2, 2, 0.5])]],
            [[np.array([2, 2, 2, 0.5])], [np.array([2, 2, 2, 0.2])]]
        ]
        slices_out = [
            [8, 8, 8, 0.5],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.5]
        ]
        fused = fuse_slices(slices_in, slice_size_h=20, slice_size_w=20, overlap=5, image_h=30, image_w=30,
                            prediction_threshold=0.4)
        self.assertEqual(fused, slices_out)

    def test_fuse_slices_2(self):
        slices_in = [
            [[np.array([2, 2, 2, 0.6])], [np.array([3, 2, 2, 0.5])]],
            [[np.array([2, 2, 2, 0.5])], [np.array([2, 2, 2, 0.2]), np.array([4, 4, 5, 0.8]), np.array([-2, -2, 5, 0.8])]]
        ]
        slices_out = [
            [8, 8, 8, 0.6],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.5],
            [26, 26, 20, 0.8],
            [2, 2, 20, 0.8]
        ]
        fused = fuse_slices(slices_in, slice_size_h=20, slice_size_w=20, overlap=5, image_h=30, image_w=30,
                            prediction_threshold=0.4)
        self.assertEqual(fused, slices_out)

    def test_fuse_slices_3(self):
        slices_in = [
            [[np.array([2, 2, 2, 0.6])], [np.array([3, 2, 2, 0.5])]],
            [[np.array([2, 2, 2, 0.5])], [np.array([2, 2, 2, 0.2]), np.array([4, 4, 5, 0.8]), np.array([-2, -2, 5, 0.8])]]
        ]
        slices_out = [
            [8, 8, 8, 0.6],
            [27, 8, 8, 0.5],
            [8, 23, 8, 0.5],
            [31, 31, 20, 0.8],
            [7, 7, 20, 0.8]
        ]
        fused = fuse_slices(slices_in, slice_size_h=20, slice_size_w=20, overlap=5, image_h=35, image_w=35,
                            prediction_threshold=0.4)
        self.assertEqual(fused, slices_out)

    def test_fuse_slices_4(self):
        slices_in = [
            [[np.array([2, 2, 2, 0.6])], [np.array([3, 2, 2, 0.5])], []],
            [[np.array([2, 2, 2, 0.5])], [np.array([2, 2, 2, 0.2]), np.array([4, 4, 5, 0.8]), np.array([-2, -2, 5, 0.8])], []]
        ]
        slices_out = [
            [8, 8, 8, 0.6],
            [31, 31, 20, 0.8],
            [7, 7, 20, 0.8]
        ]
        fused = fuse_slices(slices_in, slice_size_h=20, slice_size_w=20, overlap=5, image_h=35, image_w=50,
                            prediction_threshold=0.5)
        self.assertEqual(fused, slices_out)


    def test_nms_circles_1(self):
        circles_in = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.49],
            [8, 16, 2, 0.48]
        ]
        circles_out = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.49]
        ]
        circles_nms = nms_circles(circles_in, radius_threshold=1.0)
        self.assertEqual(circles_out, circles_nms)

    def test_nms_circles_2(self):
        circles_in = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.49],
            [0, 0, 4, 0.48]
        ]
        circles_out = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.49],
            [0, 0, 4, 0.48]
        ]
        circles_nms = nms_circles(circles_in, radius_threshold=1.5)
        self.assertEqual(circles_out, circles_nms)

    def test_nms_circles_3(self):
        circles_in = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 18, 8, 0.49],
            [8, 16, 2, 1]
        ]
        circles_out = [
            [8, 8, 8, 0.51],
            [22, 8, 8, 0.5],
            [8, 16, 2, 1]
        ]
        circles_nms = nms_circles(circles_in, radius_threshold=1.5)
        self.assertEqual(circles_out, circles_nms)

    def test_nms_circles_4(self):
        circles_in = [
            [8, 8, 8, 0.51],
            [8, 8, 8, 0.51],
        ]
        circles_out = [
            [8, 8, 8, 0.51]
        ]
        circles_nms = nms_circles(circles_in, radius_threshold=1.5)
        self.assertEqual(circles_out, circles_nms)


    def test_compute_overlap_bboxes_1(self):
        bbox1 = [10, 10, 20, 20]
        bbox2 = [10, 15, 20, 20]
        overlap = bb_intersection_over_union(bbox1, bbox2)
        self.assertEqual(0.5, overlap)

    def test_compute_overlap_bboxes_2(self):
        bbox1 = [10, 10, 20, 20]
        bbox2 = [10, 10, 20, 20]
        overlap = bb_intersection_over_union(bbox1, bbox2)
        self.assertEqual(1.0, overlap)

    def test_compute_overlap_bboxes_3(self):
        bbox1 = [10, 10, 20, 20]
        bbox2 = [20, 20, 30, 30]
        overlap = bb_intersection_over_union(bbox1, bbox2)
        self.assertEqual(0.0, overlap)

    def test_compute_overlap_bboxes_4(self):
        bbox1 = [10, 10, 20, 20]
        bbox2 = [15, 15, 25, 25]
        overlap = bb_intersection_over_union(bbox1, bbox2)
        self.assertAlmostEqual(1/7, overlap, delta=0.001)


    def test_compute_tp_fp_fn_1(self):
        predictions1 = [
            [10, 10, 10],
            [10, 15, 10],
            [30, 30, 15],
            [25, 25, 15]
        ]
        predictions2 = [
            [10, 10, 10],
            [10, 15, 7],
            [25, 25, 15],
            [100, 100, 1]
        ]
        tp, fp, fn = compute_tp_fp_fn(predictions1, predictions2, overlap_threshold=0.5)
        self.assertEqual(tp, 2)
        self.assertEqual(fp, 2)
        self.assertEqual(fn, 2)

    def test_compute_tp_fp_fn_2(self):
        predictions1 = [
            [10, 10, 10],
            [10, 15, 10],
            [30, 30, 15],
            [25, 25, 15]
        ]
        predictions2 = [
            [10, 10, 10],
            [10, 15, 10],
            [27, 27, 15]
        ]
        tp, fp, fn = compute_tp_fp_fn(predictions1, predictions2, overlap_threshold=0.5)
        self.assertEqual(tp, 3)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 0)

    def test_compute_tp_fp_fn_3(self):
        predictions1 = [
            [200, 200, 10],
        ]
        predictions2 = [
            [10, 10, 10],
            [10, 15, 7],
            [25, 25, 15],
            [100, 100, 1]
        ]
        tp, fp, fn = compute_tp_fp_fn(predictions1, predictions2, overlap_threshold=0.5)
        self.assertEqual(tp, 0)
        self.assertEqual(fp, 1)
        self.assertEqual(fn, 4)