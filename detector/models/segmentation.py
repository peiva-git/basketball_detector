import keras
from keras_segmentation.models.pspnet import pspnet_50
from keras_segmentation.models.unet import mobilenet_unet


class PSPNet50:
    def __init__(self, output_channels: int = 2, image_size: (int, int) = (1080, 1920)):
        self.__model = pspnet_50(n_classes=output_channels, input_height=image_size[0], input_width=image_size[1])
        self.__model_name = 'pspnet'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class MobileNetUNet:
    def __init__(self, output_channels: int = 2, input_shape: (int, int, int) = (224, 224, 3)):
        self.__model = mobilenet_unet(n_classes=output_channels,
                                      input_height=input_shape[0],
                                      input_width=input_shape[1])
        self.__model_name = 'mobilenet_unet'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class UNet:
    def __init__(self, input_shape: (int, int, int) = (1024, 2048, 3), number_of_classes: int = 2):
        inputs = keras.Input(shape=input_shape)
        x = keras.layers.Rescaling(1. / 255, input_shape=input_shape)(inputs)

        # [First half of the network: downsampling inputs] ###

        # Entry block
        x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        # Blocks 1, 2, 3 are identical apart from the feature depth.
        for filters in [64, 128, 256]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # [Second half of the network: upsampling inputs] ###

        for filters in [256, 128, 64, 32]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.UpSampling2D(2)(x)

            # Project residual
            residual = keras.layers.UpSampling2D(2)(previous_block_activation)
            residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
            x = keras.layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        # Add a per-pixel classification layer
        outputs = keras.layers.Conv2D(number_of_classes, 3, activation="softmax", padding="same")(x)

        self.__model = keras.Model(inputs, outputs)
        self.__model_name = 'unet'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name
