import albumentations as alb
import random

import cv2
import numpy as np


def augment_images(images, annotations):
    for i, image in enumerate(images):
        images[i], annotations[i]['bboxes'], annotations[i]['labels'] = __augment_image(image, annotations[i]['bboxes'],
                                                                                    annotations[i]['labels'])
    return images, annotations


def __augment_image(image, bboxes, labels):
    bbox_params = alb.BboxParams(format='pascal_voc', label_fields=['labels'])
    augmentations = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        #alb.Rotate(180, p=0.5),
        #alb.RandomScale(scale_limit=0.2, p=0.25),
        alb.RGBShift(r_shift_limit=15, g_shift_limit=25, b_shift_limit=20),
        alb.RandomBrightnessContrast(p=0.5),
        alb.RandomGamma(p=0.5),
        alb.OneOf([
            alb.GaussianBlur(blur_limit=9, always_apply=True),
            alb.MedianBlur(blur_limit=5, always_apply=True),
            alb.MotionBlur(blur_limit=7, always_apply=True)
        ], p=0.3),
        alb.GaussNoise(var_limit=(10, 100), p=0.25),
        alb.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT,
                             p=0.5)
    ], bbox_params=bbox_params)

    data = {'image': image, 'labels': labels, 'bboxes': bboxes}
    augmented = augmentations(**data)
    image, labels, bboxes = augmented['image'], np.array(augmented['labels']), np.array(augmented['bboxes'])
    return image, bboxes, labels
