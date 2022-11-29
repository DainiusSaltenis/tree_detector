import random

from generators.common import Generator
import os
import numpy as np
from six import raise_from
import cv2
import json
import albumentations as a

circle_classes = {
    'tree': 0
}


class CircleGenerator(Generator):
    def __init__(
            self,
            data_dir,
            set_name,
            classes=circle_classes,
            image_extension='.jpg',
            skip_truncated=False,
            skip_difficult=False,
            **kwargs
    ):
        self.data_dir = data_dir
        self.set_name = set_name
        self.classes = classes
        with open(os.path.join(data_dir, "annotations.json")) as f:
            self.image_data = json.load(f)
        self.image_names = [o for o in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, o))]
        self.image_names = self.image_names[:len(self.image_names) - 2]
        self.image_extension = image_extension
        self.__load_annotations()
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        super(CircleGenerator, self).__init__(**kwargs)

    def size(self):
        """
        Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """
        Number of classes in the dataset.
        """
        return len(self.classes)

    def has_label(self, label):
        """
        Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """
        Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """
        Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """
        Map label to name.
        """
        return self.labels[label]

    def image_aspect_ratio(self, image_index):
        """
        Compute the aspect ratio for an image with image_index.
        """
        path = os.path.join(self.data_dir, 'JPEGImages', self.image_names[image_index] + self.image_extension)
        image = cv2.imread(path)
        h, w = image.shape[:2]
        return float(w) / float(h)

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        path = os.path.join(self.data_dir, self.image_names[image_index])
        #print(self.image_names)
        image = cv2.imread(path)
        return image

    def crop_input(self, image, annotations):
        h, w, c = image.shape
        if h < self.input_size or w < self.input_size:
            return image, annotations
        min_l = min(h, w)
        scale_variance = (0.5, 2.0)
        scale_boundaries = (scale_variance[0] * self.input_size, scale_variance[1] * self.input_size, min_l)
        scale = random.randint(int(scale_boundaries[0]), int(scale_boundaries[1]))
        crop = a.Compose([a.RandomCrop(scale, scale, always_apply=True)],
                         a.BboxParams(format='pascal_voc', label_fields=['labels']))
        data = {'image': image, 'labels': annotations['labels'], 'bboxes': annotations['bboxes']}
        cropped = crop(**data)
        image, annotations['labels'], annotations['bboxes'] = cropped['image'], np.array(cropped['labels']),\
                                                              np.array(cropped['bboxes'])
        h_a, w_a, _ = image.shape
        scale_h, scale_w = h_a / self.input_size, w_a / self.input_size
        image = cv2.resize(image, dsize=(self.input_size, self.input_size))
        for bbox in annotations['bboxes']:
            bbox[0], bbox[1], bbox[2], bbox[3] = bbox[0] / scale_w, bbox[1] / scale_h, \
                                                 bbox[2] / scale_w, bbox[3] / scale_h
        return image, annotations

    def __load_annotations(self):
        self.annotations = []
        for image_name in self.image_names:
            for annotation in self.image_data:
                if annotation['filename'] == image_name:
                    bboxes = []
                    for obj in annotation['objs']:
                        circle = []
                        circle.append(obj['circle'][0] - obj['circle'][2])
                        circle.append(obj['circle'][1] - obj['circle'][2])
                        circle.append(obj['circle'][0] + obj['circle'][2])
                        circle.append(obj['circle'][1] + obj['circle'][2])
                        bboxes.append(circle)
                    self.annotations.append(bboxes)

    def load_annotations(self, image_index):
        annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}

        for annotation in self.annotations[image_index]:
            annotations['labels'] = np.concatenate([annotations['labels'], np.asarray([0])], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [np.asarray(annotation)]], axis=0)

        return annotations




