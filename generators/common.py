import cv2
import tensorflow.keras
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import warnings

from generators.utils import get_affine_transform, affine_transform, preprocess_transform_image
from generators.utils import gaussian_radius, draw_gaussian, gaussian_radius_2, draw_gaussian_2

from augmentor.augmentor import augment_images
from utils.image_drawing import draw_inputs_before_compute


class Generator(tensorflow.keras.utils.Sequence):
    def __init__(
            self,
            multi_scale=False,
            misc_effect=None,
            visual_effect=None,
            batch_size=1,
            group_method='random',
            shuffle_groups=True,
            input_size=512,
            max_objects=1000
    ):
        self.misc_effect = misc_effect
        self.visual_effect = visual_effect
        self.batch_size = int(batch_size)
        self.group_method = group_method
        self.shuffle_groups = shuffle_groups
        self.input_size = input_size
        self.output_size = self.input_size // 4
        self.max_objects = max_objects
        self.groups = None
        self.current_index = 0
        self.group_images()
        #if self.shuffle_groups:
        #    random.shuffle(self.groups)

    def on_epoch_end(self):
        if self.shuffle_groups:
            random.shuffle(self.groups)
        self.current_index = 0

    def size(self):
        """
        Size of the dataset.
        """
        raise NotImplementedError('size method not implemented')

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        raise NotImplementedError('num_classes method not implemented')

    def has_label(self, label):
        """
        Returns True if label is a known label.
        """
        raise NotImplementedError('has_label method not implemented')

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        raise NotImplementedError('has_name method not implemented')

    def name_to_label(self, name):
        """
        Map name to label.
        """
        raise NotImplementedError('name_to_label method not implemented')

    def label_to_name(self, label):
        """
        Map label to name.
        """
        raise NotImplementedError('label_to_name method not implemented')

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        raise NotImplementedError('image_aspect_ratio method not implemented')

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        raise NotImplementedError('load_image method not implemented')

    def load_annotations(self, image_index):
        """
        Load annotations for an image_index.
        """
        raise NotImplementedError('load_annotations method not implemented')

    def crop_input(self, image, annotations):
        """
        Crop a part from an input image.
        """
        raise NotImplementedError('crop_input method not implemented')

    def load_annotations_group(self, group):
        """
        Load annotations for all images in group.
        """
        annotations_group = [self.load_annotations(image_index) for image_index in group]

        return annotations_group

    def filter_annotations(self, image_group, annotations_group):
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            if len(annotations['bboxes']) == 0:
                continue
            invalid_indices = np.where(
                (annotations['bboxes'][:, 2] <= annotations['bboxes'][:, 0]) |
                (annotations['bboxes'][:, 3] <= annotations['bboxes'][:, 1]) |
                (annotations['bboxes'][:, 0] + (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0]) / 2 < 0) |
                (annotations['bboxes'][:, 1] + (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1]) / 2 < 0) |
                (annotations['bboxes'][:, 0] + (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0]) / 2 > image.shape[0]) |
                (annotations['bboxes'][:, 1] + (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1]) / 2 > image.shape[1])
            )[0]

            for k in annotations_group[index].keys():
                annotations_group[index][k] = np.delete(annotations[k], invalid_indices, axis=0)

        return image_group, annotations_group

    def clip_transformed_annotations(self, image_group, annotations_group, group):
        filtered_image_group = []
        filtered_annotations_group = []
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image_height = image.shape[0]
            image_width = image.shape[1]
            annotations['bboxes'][:, 0] = np.clip(annotations['bboxes'][:, 0], 0, image_width - 2)
            annotations['bboxes'][:, 1] = np.clip(annotations['bboxes'][:, 1], 0, image_height - 2)
            annotations['bboxes'][:, 2] = np.clip(annotations['bboxes'][:, 2], 1, image_width - 1)
            annotations['bboxes'][:, 3] = np.clip(annotations['bboxes'][:, 3], 1, image_height - 1)
            small_indices = np.where(
                (annotations['bboxes'][:, 2] - annotations['bboxes'][:, 0] < 10) |
                (annotations['bboxes'][:, 3] - annotations['bboxes'][:, 1] < 10)
            )[0]

            if len(small_indices):
                for k in annotations_group[index].keys():
                    annotations_group[index][k] = np.delete(annotations[k], small_indices, axis=0)

        return filtered_image_group, filtered_annotations_group

    def load_image_group(self, group):
        """
        Load images for all images in a group.
        """
        return [self.load_image(image_index) for image_index in group]

    def group_images(self):
        """
        Order the images according to self.order and makes groups of self.batch_size.
        """
        order = list(range(self.size()))
        #if self.group_method == 'random':
        #    random.shuffle(order)
        #elif self.group_method == 'ratio':
        #    order.sort(key=lambda x: self.image_aspect_ratio(x))

        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                       range(0, len(order), self.batch_size)]

    def compute_inputs(self, image_group, annotations_group):
        """
        Compute inputs for the network using an image_group.
        """
        batch_images = np.zeros((len(image_group), self.input_size, self.input_size, 3), dtype=np.float32)

        batch_hms = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                             dtype=np.float32)
        batch_hms_2 = np.zeros((len(image_group), self.output_size, self.output_size, self.num_classes()),
                               dtype=np.float32)
        batch_rs = np.zeros((len(image_group), self.max_objects, 1), dtype=np.float32)
        batch_regs = np.zeros((len(image_group), self.max_objects, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((len(image_group), self.max_objects), dtype=np.float32)
        batch_indices = np.zeros((len(image_group), self.max_objects), dtype=np.float32)

        # copy all images to the upper left part of the image batch object
        for b, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
            s = max(image.shape[0], image.shape[1]) * 1.0

            # inputs
            image = self.preprocess_image(image, c, s, tgt_w=self.input_size, tgt_h=self.input_size)
            batch_images[b] = image

            # outputs
            bboxes = annotations['bboxes']
            class_ids = annotations['labels']

            trans_output = get_affine_transform(c, s, self.output_size)
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i].copy()
                cls_id = int(class_ids[i])
                # (x1, y1)
                bbox[:2] = affine_transform(bbox[:2], trans_output)
                # (x2, y2)
                bbox[2:] = affine_transform(bbox[2:], trans_output)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_size - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_size - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                r = max(h, w) / 2
                if r > 0:
                    radius_h, radius_w = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius_h = max(0, int(radius_h))
                    radius_w = max(0, int(radius_w))

                    radius = gaussian_radius_2((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius * 1.5))
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)
                    draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius_h, radius_w)
                    draw_gaussian_2(batch_hms_2[b, :, :, cls_id], ct_int, radius)
                    batch_rs[b, i] = 1. * r
                    batch_indices[b, i] = ct_int[1] * self.output_size + ct_int[0]
                    batch_regs[b, i] = ct - ct_int
                    batch_reg_masks[b, i] = 1

        visualise = False
        if visualise:
            from utils.image_drawing import draw_inputs_computed
            draw_inputs_computed(batch_images, batch_hms_2)

        return [batch_images, batch_hms_2, batch_rs, batch_regs, batch_reg_masks, batch_indices]

    def compute_targets(self, image_group, annotations_group):
        return np.zeros((len(image_group),))

    def compute_inputs_targets(self, group):
        # load images and annotations
        image_group = self.load_image_group(group)
        annotations_group = self.load_annotations_group(group)

        # Load test visualization
        visualize = False
        if visualize:
            draw_inputs_before_compute(image_group, annotations_group)

        # clip annotations before transform to avoid bbox points being out of image shape
        # image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        image_group, annotations_group = self.crop_input_from_image_group(image_group, annotations_group)

        # Crop test visualization
        visualize = False
        if visualize:
            draw_inputs_before_compute(image_group, annotations_group)

        # check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group)

        # Augment images for regularization
        image_group, annotations_group = augment_images(image_group, annotations_group)

        # Augment test visualization
        visualize = False
        if visualize:
            draw_inputs_before_compute(image_group, annotations_group)

        # once again check validity of annotations
        image_group, annotations_group = self.filter_annotations(image_group, annotations_group)

        # Filter test visualization
        visualize = False
        if visualize:
            draw_inputs_before_compute(image_group, annotations_group)

        # check validity of annotations, clip edges
        # image_group, annotations_group = self.clip_transformed_annotations(image_group, annotations_group, group)

        inputs = self.compute_inputs(image_group, annotations_group)
        targets = self.compute_targets(image_group, annotations_group)

        return inputs, targets

    def crop_input_from_image_group(self, image_group, annotations_group):
        cropped = [self.crop_input(*input) for input in zip(image_group, annotations_group)]
        return [c[0] for c in cropped], [c[1] for c in cropped]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        group = self.groups[self.current_index]
        inputs, targets = self.compute_inputs_targets(group)
        current_index = self.current_index + 1
        if current_index >= len(self.groups):
            current_index = current_index % (len(self.groups))
        self.current_index = current_index
        return inputs, targets

    def preprocess_image(self, image, c, s, tgt_w, tgt_h):
        return preprocess_transform_image(image, c, s, tgt_w, tgt_h)

