import os
import tensorflow as tf
import keras_cv
from keras_segmentation.models.pspnet import pspnet_50


def get_model_callbacks(
        self,
        early_stop_patience: int,
        reduce_lr_patience: int) -> [tf.keras.callbacks.Callback]:
    model_dir_path = os.path.join('out', 'training-callback-results', self.__model_name)
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
        ),
        tf.keras.callbacks.ProgbarLogger()
    ]


class SimpleClassifier:
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.Sequential([
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
        self.__model_name = 'simple-classifier'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class ResNet152V2:
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = keras_cv.models.ImageClassifier(
            backbone=keras_cv.models.ResNet152V2Backbone(input_shape=(image_height, image_width, 3)),
            num_classes=number_of_classes
        )
        self.__model_name = 'resnet152v2'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class MobileNet:
    # docs at https://keras.io/api/applications/mobilenet/#mobilenetv2-function
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.applications.MobileNetV2(
            input_shape=(image_height, image_width, 3),
            weights=None,
            classes=number_of_classes
        )
        self.__model_name = 'mobilenetv2'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class EfficientNet:
    # docs at https://keras.io/api/applications/efficientnet_v2/#efficientnetv2b0-function
    def __init__(self,
                 number_of_classes: int = 2,
                 image_width: int = 50,
                 image_height: int = 50):
        self.__model = tf.keras.applications.EfficientNetV2B0(
            weights=None,
            classes=number_of_classes
        )
        self.__model_name = 'efficientnetv2b0'


class Segmentation:
    def __init__(self, output_channels: int = 2, image_size: (int, int) = (1080, 1920)):
        self.__model = pspnet_50(n_classes=output_channels, input_height=image_size[0], input_width=image_size[1])
        self.__model_name = 'pspnet'
        self.__model.summary()

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name
