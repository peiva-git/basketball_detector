import pathlib
import os.path
from typing import Any

import tensorflow as tf
import numpy as np


class DatasetBuilder:
    def __init__(self, data_directory: str, validation_percentage: float = 0.2):
        data_path = pathlib.Path(data_directory)
        self.image_count = len(list(data_path.glob('/*/*/*/*.png')))
        list_dataset = tf.data.Dataset.list_files(str(data_path / '*/*/*/*'), shuffle=False)
        list_dataset = list_dataset.shuffle(self.image_count, reshuffle_each_iteration=False)
        self.class_names = np.array(sorted([item.name for item in data_path.glob('/*/*/*')]))
        print('Found the following classes: ', self.class_names)

        validation_size = int(self.image_count * validation_percentage)
        self.train_dataset = list_dataset.skip(validation_size)
        self.validation_dataset = list_dataset.take(validation_size)
        print(tf.data.experimental.cardinality(self.train_dataset).numpy(), ' images in training dataset')
        print(tf.data.experimental.cardinality(self.validation_dataset).numpy(), ' images in validation dataset')

        self.train_dataset = self.train_dataset.map(
            self._get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        self.validation_dataset = self.validation_dataset.map(
            self._get_image_label_pair_from_path,
            num_parallel_calls=tf.data.AUTOTUNE
        )

    def _get_label(self, file_path: str) -> int:
        parts = tf.strings.split(file_path, os.path.sep)
        one_hot = parts[-2] == self.class_names
        return tf.argmax(one_hot)

    @staticmethod
    def decode_image(image_data, image_width: int = 50, image_height: int = 50):
        image = tf.io.decode_png(image_data, channels=3)
        return tf.image.resize(image, [image_width, image_height])

    def _get_image_label_pair_from_path(self, file_path) -> (Any, int):
        label = self._get_label(file_path)
        image_data = tf.io.read_file(file_path)
        image = self.decode_image(image_data)
        return image, label

    @staticmethod
    def _configure_for_performance(dataset: tf.data.Dataset, buffer_size: int, batch_size: int) -> tf.data.Dataset:
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def configure_datasets_for_performance(self, buffer_size: int = 1000, batch_size: int = 32):
        self.train_dataset = self._configure_for_performance(self.train_dataset, buffer_size, batch_size)
        self.validation_dataset = self._configure_for_performance(self.validation_dataset, buffer_size, batch_size)

    def build(self) -> (tf.data.Dataset, tf.data.Dataset):
        return self.train_dataset, self.validation_dataset
