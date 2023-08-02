import pathlib
import os.path
from typing import Any

import tensorflow as tf
import numpy as np


def decode_image(image_data, image_width: int = 50, image_height: int = 50, channels: int = 3):
    image = tf.io.decode_png(image_data, channels=channels)
    return tf.image.resize(image, [image_height, image_width])


def configure_for_performance(dataset: tf.data.Dataset, buffer_size: int, batch_size: int) -> tf.data.Dataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


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

    def __get_label(self, file_path: tf.Tensor) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.__class_names
        return tf.argmax(one_hot)

    def __get_image_label_pair_from_path(self, file_path: tf.Tensor) -> (Any, int):
        label = self.__get_label(file_path)
        image_data = tf.io.read_file(file_path)
        image = decode_image(image_data)
        return image, label

    def configure_datasets_for_performance(self, buffer_size: int = tf.data.AUTOTUNE, batch_size: int = 32):
        self.__train_dataset = configure_for_performance(self.__train_dataset, buffer_size, batch_size)
        self.__validation_dataset = configure_for_performance(self.__validation_dataset, buffer_size, batch_size)


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
        print(len(input_image_paths), 'frames, with', len(mask_image_paths), 'corresponding ground truth masks')
        print(number_of_validation_samples, 'samples will be set aside as validation data')

        if len(input_image_paths) != len(mask_image_paths):
            raise ValueError('The number of frames is different than the number of ground truth masks, aborting')

        input_image_paths.sort()
        mask_image_paths.sort(key=lambda file_path: file_path.split('_')[-1].split('.')[-2])
        samples_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(input_image_paths))
        masks_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(mask_image_paths))

        samples_dataset = samples_dataset.map(
            self.__get_frame_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        masks_dataset = masks_dataset.map(
            self.__get_mask_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = tf.data.Dataset.zip((samples_dataset, masks_dataset))
        dataset.shuffle(len(mask_image_paths), reshuffle_each_iteration=False)
        self.__train_dataset = dataset.skip(number_of_validation_samples)
        self.__validation_dataset = dataset.take(number_of_validation_samples)
        print(tf.data.experimental.cardinality(self.__train_dataset).numpy(), 'frames in training dataset')
        print(tf.data.experimental.cardinality(self.__validation_dataset).numpy(), 'frames in validation dataset')

    @staticmethod
    def __get_frame_from_path(filepath: tf.Tensor):
        image_data = tf.io.read_file(filepath)
        image = decode_image(image_data, image_width=1024, image_height=512)
        return image

    @staticmethod
    def __get_mask_from_path(filepath: tf.Tensor):
        image_data = tf.io.read_file(filepath)
        image = decode_image(image_data, image_width=1024, image_height=512, channels=1)
        return image

    @property
    def train_dataset(self) -> tf.data.Dataset:
        return self.__train_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        return self.__validation_dataset

    @property
    def number_of_samples(self) -> int:
        return tf.data.experimental.cardinality(self.__train_dataset).numpy() + \
               tf.data.experimental.cardinality(self.__validation_dataset).numpy()

    def configure_datasets_for_performance(self, buffer_size: int = tf.data.AUTOTUNE, batch_size: int = 10):
        self.__train_dataset = configure_for_performance(self.__train_dataset, buffer_size, batch_size)
        self.__validation_dataset = configure_for_performance(self.__validation_dataset, buffer_size, batch_size)
