import os

import tensorflow as tf
import keras_cv

# Simple classifier implemented following the tutorial at
# https://www.tensorflow.org/tutorials/images/classification#a_basic_keras_model


def get_model_callbacks(
        model_name: str,
        early_stop_patience: int,
        reduce_lr_patience: int) -> [tf.keras.callbacks.Callback]:
    model_dir_path = os.path.join('out', 'training-callback-results', model_name)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir_path, 'checkpoint'),
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=os.path.join(model_dir_path, 'backup')),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=early_stop_patience,
            start_from_epoch=2
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir_path, 'tensorboard-logs'),
            histogram_freq=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=reduce_lr_patience,
            min_lr=0.001
        )
    ]


class SimpleClassifier(tf.keras.Sequential):
    def __init__(self, number_of_classes: int = 2, image_width: int = 50, image_height: int = 50):
        self.model = super().__init__([
            tf.keras.layers.Rescaling(1. / 255, input_shape=(image_width, image_height, 3)),
            tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(number_of_classes)
        ])


class ResNetClassifier:
    def __init__(self):
        self.model = keras_cv.models.ImageClassifier(
            backbone=keras_cv.models.ResNet152V2Backbone(),
            num_classes=2
        )