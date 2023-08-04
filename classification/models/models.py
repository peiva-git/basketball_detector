import os
import tensorflow as tf
import keras_cv
from tensorflow_examples.models.pix2pix import pix2pix


class Classifier:
    def __init__(self,
                 number_of_classes: int = 2,
                 model_name: str = 'simple-classifier',
                 image_width: int = 50,
                 image_height: int = 50):
        if model_name == 'simple-classifier':
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
        else:
            self.__model = keras_cv.models.ImageClassifier(
                backbone=keras_cv.models.ResNet152V2Backbone(input_shape=(image_width, image_height, 3)),
                num_classes=number_of_classes
            )
        self.__model_name = model_name

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name

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


class Segmentation:
    def __init__(self, output_channels: int = 2, image_size: (int, int) = (1080, 1920)):
        base_model = tf.keras.applications.MobileNetV2(input_shape=image_size + (3,), include_top=False)

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',  # 64x64
            'block_3_expand_relu',  # 32x32
            'block_6_expand_relu',  # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',  # 4x4
        ]
        base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

        down_stack.trainable = False

        up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),  # 32x32 -> 64x64
        ]

        inputs = tf.keras.layers.Input(shape=[128, 128, 3])

        # Downsampling through the model
        skips = down_stack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            filters=output_channels, kernel_size=3, strides=2,
            padding='same')  # 64x64 -> 128x128

        x = last(x)

        self.__model = tf.keras.Model(inputs=inputs, outputs=x)
        self.__model_name = 'segmenter'
        self.__model.summary()

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name
