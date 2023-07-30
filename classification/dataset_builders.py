import pathlib
import os.path
import random
from typing import Any

import tensorflow as tf
import numpy as np


def decode_image(image_data, image_width: int = 50, image_height: int = 50):
    image = tf.io.decode_png(image_data, channels=3)
    return tf.image.resize(image, [image_width, image_height])


class ClassificationDatasetBuilder:
    def __init__(self, data_directory: str, validation_percentage: float = 0.2, reduce_percentage: float = 0.0):
        data_path = pathlib.Path(data_directory)
        self.__image_count = len(list(data_path.glob('*/*/*/*.png')))
        list_dataset = tf.data.Dataset.list_files(str(data_path / '*/*/*/*'), shuffle=False)
        list_dataset = list_dataset.shuffle(self.__image_count, reshuffle_each_iteration=False)
        self.__class_names = np.unique(sorted([item.name for item in data_path.glob('*/*/*')]))
        print('Found the following classes: ', self.__class_names)

        validation_size = int(self.__image_count * validation_percentage)
        self.__train_dataset = list_dataset.skip(validation_size)
        self.__validation_dataset = list_dataset.take(validation_size)
        print(tf.data.experimental.cardinality(self.__train_dataset).numpy(), 'images in training dataset')
        print(tf.data.experimental.cardinality(self.__validation_dataset).numpy(), 'images in validation dataset')

        if reduce_percentage != 0.0:
            print('Reducing both datasets by ', reduce_percentage * 100, '%...')
            self.__train_dataset = self.__train_dataset \
                .skip(int(tf.data.experimental.cardinality(self.__train_dataset).numpy() * reduce_percentage))
            self.__validation_dataset = self.__validation_dataset \
                .skip(int(tf.data.experimental.cardinality(self.__validation_dataset).numpy() * reduce_percentage))
            print(tf.data.experimental.cardinality(self.__train_dataset).numpy(), 'images in training dataset')
            print(tf.data.experimental.cardinality(self.__validation_dataset).numpy(), 'images in validation dataset')

        self.__train_dataset = self.__train_dataset.map(
            self.__get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.__validation_dataset = self.__validation_dataset.map(
            self.__get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    @property
    def number_of_images(self) -> int:
        return self.__image_count

    @property
    def class_names(self) -> [str, str]:
        return self.__class_names

    @property
    def train_dataset(self) -> tf.data.Dataset:
        return self.__train_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        return self.__validation_dataset

    def __get_label(self, file_path: str) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.__class_names
        return tf.argmax(one_hot)

    def __get_image_label_pair_from_path(self, file_path: str) -> (Any, int):
        label = self.__get_label(file_path)
        image_data = tf.io.read_file(file_path)
        image = decode_image(image_data)
        return image, label

    @staticmethod
    def __configure_for_performance(dataset: tf.data.Dataset, buffer_size: int, batch_size: int) -> tf.data.Dataset:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def configure_datasets_for_performance(self, buffer_size: int = 1000, batch_size: int = 32):
        self.__train_dataset = self.__configure_for_performance(self.__train_dataset, buffer_size, batch_size)
        self.__validation_dataset = self.__configure_for_performance(self.__validation_dataset, buffer_size, batch_size)


class SegmentationDatasetBuilder:
    def __init__(self,
                 data_directory: str,
                 masks_directory: str,
                 validation_percentage: float = 0.2):
        input_image_paths = [
            os.path.join(data_directory, filename)
            for filename in os.listdir(data_directory)
            if filename.endswith('.png')
        ]
        mask_image_paths = [
            os.path.join(masks_directory, filename)
            for filename in os.listdir(masks_directory)
            if filename.endswith('.png')
        ]
        number_of_validation_samples = int(len(input_image_paths) * validation_percentage)

        input_image_paths.sort()
        input_mask_path_pairs = []
        for mask_path in mask_image_paths:
            frame_number = int(mask_path.split('_')[-1].split('.')[-2])
            input_mask_path_pairs.append((input_image_paths[frame_number - 1], mask_path))

        random.Random(2023).shuffle(input_mask_path_pairs)
        self.__train_input_mask_path_pairs = input_mask_path_pairs[:-number_of_validation_samples]
        self.__validation_input_mask_path_pairs = input_mask_path_pairs[-number_of_validation_samples:]

    @property
    def train_input_mask_path_pairs(self) -> [(str, str)]:
        return self.__train_input_mask_path_pairs

    @property
    def validation_input_mask_path_pairs(self) -> [(str, str)]:
        return self.__validation_input_mask_path_pairs


class Basketballs(tf.keras.utils.Sequence):
    def __init__(self,
                 input_mask_path_pairs: [(str, str)],
                 batch_size: int = 1000,
                 image_size: (int, int) = (1920, 1080)):
        self.__batch_size = batch_size
        self.__image_size = image_size
        self.__input_mask_path_pairs = input_mask_path_pairs

    def __getitem__(self, index) -> ([], []):
        i = index * self.__batch_size
        batch_input_mask_path_pairs = self.__input_mask_path_pairs[i: i + self.__batch_size]
        x = np.zeros((self.__batch_size,) + self.__image_size + (3,), dtype='float32')
        y = np.zeros((self.__batch_size,) + self.__image_size + (1,), dtype='uint8')
        for j, input_path, mask_path in enumerate(batch_input_mask_path_pairs):
            frame_image = tf.keras.utils.load_img(input_path, target_size=self.__image_size)
            x[j] = frame_image
            mask_image = tf.keras.utils.load_img(mask_path, target_size=self.__image_size, color_mode='grayscale')
            y[j] = np.expand_dims(mask_image, 2)
            y[j] -= 1
        return x, y

    def __len__(self):
        return len(self.__input_mask_path_pairs) // self.__batch_size
