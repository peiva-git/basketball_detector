import tensorflow as tf

# Simple classifier implemented following the tutorial at
# https://www.tensorflow.org/tutorials/images/classification#a_basic_keras_model


class SimpleClassifier(tf.keras.Sequential):
    def __init__(self, number_of_classes: int = 2, image_width: int = 50, image_height: int = 50):
        super().__init__([
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
