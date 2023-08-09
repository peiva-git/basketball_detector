from keras_segmentation.models.pspnet import pspnet_50


class Segmentation:
    def __init__(self, output_channels: int = 2, image_size: (int, int) = (1080, 1920)):
        self.__model = pspnet_50(n_classes=output_channels, input_height=image_size[0], input_width=image_size[1])
        self.__model_name = 'pspnet'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name
