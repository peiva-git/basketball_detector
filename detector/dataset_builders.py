import glob
import math
import pathlib
import os.path
from typing import Any
from random import shuffle

import keras_cv.layers
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
        print(self.__train_dataset.cardinality().numpy(), 'images in training dataset')
        print(self.__validation_dataset.cardinality().numpy(), 'images in validation dataset')

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
        image = decode_image(image_data, image_width=112, image_height=112)
        return image, label

    def configure_datasets_for_performance(self, shuffle_buffer_size: int = 10000, input_batch_size: int = 32):
        self.__train_dataset = configure_for_performance(self.__train_dataset, shuffle_buffer_size, input_batch_size)
        self.__validation_dataset = self.__validation_dataset.batch(input_batch_size)


class ClassificationSequenceBuilder:
    def __init__(self, data_directory: str, batch_size: int, validation_percentage: float = 0.2):
        data_path = pathlib.Path(data_directory)
        print('Gathering all image paths...')
        image_paths = [
            image_path
            for image_path in glob.iglob(str(data_path / '*/*/*/*.png'))
        ]
        shuffle(image_paths)
        print(f'Found {len(image_paths)} images')
        validation_size = int(len(image_paths) * validation_percentage)
        validation_paths = image_paths[:validation_size]
        training_paths = image_paths[validation_size:]
        print(f'{len(validation_paths)} images in validation dataset')
        print(f'{len(training_paths)} images in training dataset')
        self.__training_sequence = ClassificationSequence(data_directory, training_paths, batch_size)
        self.__validation_sequence = ClassificationSequence(data_directory, validation_paths, batch_size)

    @property
    def training_sequence(self):
        return self.__training_sequence

    @property
    def validation_sequence(self):
        return self.__validation_sequence


class ClassificationSequence(tf.keras.utils.Sequence):
    def __init__(self, data_directory: str, images_paths: list[str], batch_size: int):
        data_path = pathlib.Path(data_directory)
        self.__batch_size = batch_size
        self.__image_paths = images_paths
        self.__class_names = np.unique(sorted([item.name for item in data_path.glob('*/*/*')]))
        print(f'Found classes {self.__class_names}')

    def __getitem__(self, index):
        low = index * self.__batch_size
        high = min(low + self.__batch_size, len(self.__image_paths))
        batch_paths = self.__image_paths[low:high]
        batch_labels = [
            self.__get_label(tf.constant(image_path))
            for image_path in batch_paths
        ]
        batch_images = [
            self.__get_image(tf.constant(image_path))
            for image_path in batch_paths
        ]
        return tf.stack(batch_images), tf.stack(batch_labels)

    def __len__(self):
        return math.ceil(len(self.__image_paths) / self.__batch_size)

    def __get_label(self, file_path: tf.Tensor) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.__class_names
        return tf.one_hot(tf.argmax(one_hot), 2)

    @staticmethod
    def __get_image(file_path: tf.Tensor):
        image_data = tf.io.read_file(file_path)
        return decode_image(image_data, image_width=112, image_height=112, channels=3)


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
            match_directory = pathlib.Path(match_directory_path)
            match_input_image_paths = [
                match_input_image_path
                for match_input_image_path in glob.iglob(str(match_directory / 'frames/*.png'))
            ]
            match_mask_image_paths = [
                match_mask_image_path
                for match_mask_image_path in glob.iglob(str(match_directory / 'masks/*.png'))
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

        filenames_dataset = tf.data.Dataset.zip((samples, masks))
        filenames_dataset = filenames_dataset.shuffle(buffer_size=filenames_dataset.cardinality(),
                                                      reshuffle_each_iteration=False)
        dataset = filenames_dataset.map(
            self.__get_frame_and_mask_from_filepaths,
            num_parallel_calls=tf.data.AUTOTUNE
        )

        print(f'Found {dataset.cardinality().numpy()} frames in total')
        self.__train_dataset = dataset.skip(validation_size)
        self.__validation_dataset = dataset.take(validation_size)
        print(f'{self.__train_dataset.cardinality().numpy()} frames in training dataset')
        print(f'{self.__validation_dataset.cardinality().numpy()} frames in validation dataset')

    @staticmethod
    def __get_frame_and_mask_from_filepaths(frame_filepath: tf.Tensor, mask_filepath: tf.Tensor):
        frame_data = tf.io.read_file(frame_filepath)
        mask_data = tf.io.read_file(mask_filepath)
        frame = decode_image(frame_data, image_width=1024, image_height=512)
        mask = decode_image(mask_data, image_width=1024, image_height=512, channels=1)
        # 2 as the number of classes
        mask = tf.one_hot(tf.cast(mask, tf.uint8), 2)
        mask = tf.squeeze(mask)
        mask = tf.cast(mask, tf.float32)
        return frame, mask

    def augment_train_dataset(self):
        self.__train_dataset = self.__train_dataset.map(
            SegmentationDatasetAugmentor(2023),
            num_parallel_calls=tf.data.AUTOTUNE
        )

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


class SegmentationDatasetAugmentor(tf.keras.layers.Layer):
    def __init__(self, seed: int):
        super().__init__()
        self.__augment_inputs = tf.keras.Sequential([
            keras_cv.layers.RandomFlip(rate=1.0, seed=seed),
            keras_cv.layers.RandomCrop(height=1024, width=2048, seed=seed),
        ])
        self.__augment_masks = tf.keras.Sequential([
            keras_cv.layers.RandomFlip(rate=1.0, seed=seed),
            keras_cv.layers.RandomCrop(height=1024, width=2048, seed=seed),
        ])

    def call(self, inputs, masks, **kwargs):
        inputs = self.__augment_inputs(inputs)
        masks = self.__augment_masks(masks)
        return inputs, masks
