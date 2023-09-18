import random

import cv2 as cv
import numpy as np


def draw_crop_boundaries_on_frame(image: np.ndarray, crops: list[(int, int, np.ndarray)]):
    for i, (crop_x, crop_y, crop) in enumerate(crops):
        if i == 0:
            cv.rectangle(
                image,
                (crop_x, crop_y),
                (crop_x + crop.shape[1], crop_y + crop.shape[0]),
                color=(0, 0, 255),
                thickness=5
            )
        else:
            cv.rectangle(
                image,
                (crop_x, crop_y),
                (crop_x + crop.shape[1], crop_y + crop.shape[0]),
                color=(0, 255, 0),
                thickness=5
            )


def generate_random_crops(image: np.ndarray, number_of_crops: int, variance: int) -> [int, int, np.ndarray]:
    image_height, image_width, _ = image.shape
    crops = []
    crop_width = int(image_width * 0.9)
    crop_height = int(image_height * 0.9)
    first_crop_x, first_crop_y, first_crop = __get_random_crop(image, crop_height, crop_width)
    crops.append((first_crop_x, first_crop_y, first_crop))
    for i in range(1, number_of_crops):
        x = random.randint(first_crop_x - variance, first_crop_x + variance)
        y = random.randint(first_crop_y - variance, first_crop_y + variance)
        crop = image[
            max(0, y):min(y + crop_height, image_height),
            max(0, x):min(x + crop_width, image_width)
        ]
        crops.append((max(0, x), max(0, y), crop))
    return crops


def __get_random_crop(image: np.ndarray, crop_height: int, crop_width: int) -> (int, int, np.ndarray):
    image_height, image_width, _ = image.shape
    max_x = image_width - crop_width
    max_y = image_height - crop_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return x, y, image[y:y + crop_height, x:x + crop_width]
