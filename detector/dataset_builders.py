import glob
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
    def __init__(self, data_directory: str, validation_percentage: float = 0.2):
        data_path = pathlib.Path(data_directory)
        all_images_dataset = tf.data.Dataset.list_files(str(data_path / '*/*/*/*'), seed=2023)
        self.__image_count = tf.data.experimental.cardinality(all_images_dataset).numpy()
        self.__class_names = np.unique(sorted([item.name for item in data_path.glob('*/*/*')]))
        print('Found the following classes: ', self.__class_names)

        validation_size = int(self.__image_count * validation_percentage)
        self.__train_dataset = all_images_dataset.skip(validation_size)
        self.__validation_dataset = all_images_dataset.take(validation_size)
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

    def configure_datasets_for_performance(self, shuffle_buffer_size: int = 10000, input_batch_size: int = 32):
        self.__train_dataset = configure_for_performance(self.__train_dataset, shuffle_buffer_size, input_batch_size)
        self.__validation_dataset = self.__validation_dataset.batch(input_batch_size)


class SegmentationDatasetBuilder:
    def __init__(self,
                 data_directory: str,
                 validation_percentage: float = 0.2):
        data_path = pathlib.Path(data_directory)
        input_image_paths = [
            input_image_path
            for input_image_path in glob.iglob(str(data_path / '*/*/frames/*.png'))
        ]
        mask_image_paths = [
            mask_image_path
            for mask_image_path in glob.iglob(str(data_path / '*/*/masks/*.png'))
        ]
        self.__image_count = len(input_image_paths)
        validation_size = int(self.__image_count * validation_percentage)
        print(len(input_image_paths), 'frames, with', len(mask_image_paths), 'corresponding ground truth masks')
        print(validation_size, 'samples will be set aside as validation data')

        if len(input_image_paths) != len(mask_image_paths):
            raise ValueError('The number of frames is different than the number of ground truth masks, aborting')

        samples_datasets = []
        masks_datasets = []
        for match_directory_path in glob.glob(str(data_path / '*/*/')):
            match_input_image_paths = [
                match_input_image_path
                for match_input_image_path in glob.iglob(match_directory_path + 'frames/*.png')
            ]
            match_mask_image_paths = [
                match_mask_image_path
                for match_mask_image_path in glob.iglob(match_directory_path + 'masks/*.png')
            ]
            match_input_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
            match_mask_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))

            samples_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(match_input_image_paths))
            masks_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(match_mask_image_paths))
            samples_datasets.append(samples_dataset)
            masks_datasets.append(masks_dataset)

        samples = samples_datasets[0]
        for dataset in samples_datasets[1:]:
            samples = samples.concatenate(dataset)

        masks = masks_datasets[0]
        for dataset in masks_datasets[1:]:
            masks = masks.concatenate(dataset)

        samples = samples.map(
            self.__get_frame_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        masks = masks.map(
            self.__get_mask_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = tf.data.Dataset.zip((samples, masks))
        print(f'Found {dataset.cardinality().numpy()} frames in total')
        dataset = dataset.shuffle(buffer_size=dataset.cardinality(), reshuffle_each_iteration=False)
        self.__train_dataset = dataset.skip(validation_size)
        self.__validation_dataset = dataset.take(validation_size)
        print(f'{self.__train_dataset.cardinality().numpy()} frames in training dataset')
        print(f'{self.__validation_dataset.cardinality().numpy()} frames in validation dataset')

    @staticmethod
    def __get_frame_from_path(filepath: tf.Tensor):
        image_data = tf.io.read_file(filepath)
        image = decode_image(image_data, image_width=1920, image_height=1080)
        return image

    @staticmethod
    def __get_mask_from_path(filepath: tf.Tensor):
        image_data = tf.io.read_file(filepath)
        image = decode_image(image_data, image_width=1920, image_height=1080, channels=1)
        return image

    @property
    def train_dataset(self) -> tf.data.Dataset:
        return self.__train_dataset

    @property
    def validation_dataset(self) -> tf.data.Dataset:
        return self.__validation_dataset

    @property
    def number_of_samples(self) -> int:
        return self.__train_dataset.cardinality().numpy() + self.__validation_dataset.cardinality().numpy()

    def configure_datasets_for_performance(self, shuffle_buffer_size: int = 1000, input_batch_size: int = 10):
        self.__train_dataset = configure_for_performance(self.__train_dataset, shuffle_buffer_size, input_batch_size)
        self.__validation_dataset = self.__validation_dataset.batch(input_batch_size)
