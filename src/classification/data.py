import os.path
import pathlib
from typing import Any

import tensorflow as tf
import numpy as np

from .core import IMAGE_WIDTH, BATCH_SIZE
from .core import IMAGE_HEIGHT

DATA_DIR = '/mnt/DATA/tesi/dataset/dataset_classification/pallacanestro_trieste'
data_dir = pathlib.Path(DATA_DIR)
image_count = len(list(data_dir.glob('/*/*/*/*.png')))
listDataset = tf.data.Dataset.list_files(str(data_dir / '*/*/*/*'), shuffle=False)
listDataset = listDataset.shuffle(image_count, reshuffle_each_iteration=False)

classNames = np.array(sorted([item.name for item in data_dir.glob('/*/*/*')]))
print('Found the following classes: ', classNames)

validationSize = int(image_count * 0.2)
trainDataset = listDataset.skip(validationSize)
validationDataset = listDataset.take(validationSize)

print(tf.data.experimental.cardinality(trainDataset).numpy(), ' images in training dataset')
print(tf.data.experimental.cardinality(validationDataset).numpy(), ' images in validation dataset')


def get_label(file_path: str) -> int:
    parts = tf.strings.split(file_path, os.path.sep)
    one_hot = parts[-2] == classNames
    return tf.argmax(one_hot)


def decode_image(image_data):
    image = tf.io.decode_png(image_data, channels=3)
    return tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])


def process_path(file_path) -> (Any, int):
    label = get_label(file_path)
    image_data = tf.io.read_file(file_path)
    image = decode_image(image_data)
    return image, label


def configure_for_performance(dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


trainDataset = trainDataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
validationDataset = validationDataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)

trainDataset = configure_for_performance(trainDataset)
validationDataset = configure_for_performance(validationDataset)
