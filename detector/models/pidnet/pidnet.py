from keras import layers, models
import tensorflow as tf

from detector.models.pidnet.model_utils import PagFM, Light_Bag, PAPPM, DAPPPM, Bag, segmentation_head
from detector.models.pidnet.resnet import basic_block, basicblock_expansion, bottleneck_expansion, bottleneck_block

bn_mom = 0.1

"""
apply multiple RB or RBB blocks.
x_in: input tensor
block: block to apply it can be RB or RBB
inplanes: input tensor channels
planes: output tensor channels
blocks_num: number of time block to applied
stride: stride
expansion: expand last dimension
"""


def make_layer(x_in, block, inplanes, planes, blocks_num, stride=1, expansion=1):
    downsample = None
    if stride != 1 or inplanes != planes * expansion:
        downsample = layers.Conv2D((planes * expansion), kernel_size=(1, 1), strides=stride, use_bias=False)(x_in)
        downsample = layers.BatchNormalization(momentum=bn_mom)(downsample)
        # In original resnet paper relu was applied, But in pidenet it was not used
        # So commenting out for now
        # downsample = layers.Activation("relu")(downsample)

    x = block(x_in, planes, stride, downsample)
    for i in range(1, blocks_num):
        if i == (blocks_num - 1):
            x = block(x, planes, stride=1, no_relu=True)
        else:
            x = block(x, planes, stride=1, no_relu=False)

    return x


class PIDNet:
    def __init__(self,
                 input_shape=[1024, 2048, 3],
                 m=2,
                 n=3,
                 num_classes=19,
                 planes=64,
                 ppm_planes=96,
                 head_planes=128,
                 augment=True):

        x_in = layers.Input(input_shape)

        input_shape = tf.keras.backend.int_shape(x_in)
        height_output = input_shape[1] // 8
        width_output = input_shape[2] // 8

        # I Branch
        x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x_in)
        x = layers.BatchNormalization(momentum=bn_mom)(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(planes, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = layers.BatchNormalization(momentum=bn_mom)(x)
        x = layers.Activation("relu")(x)

        x = make_layer(x, basic_block, planes, planes, m, expansion=basicblock_expansion)  # layer1
        x = layers.Activation("relu")(x)

        x = make_layer(x, basic_block, planes, planes * 2, m, stride=2, expansion=basicblock_expansion)  # layer2
        x = layers.Activation("relu")(x)

        x_ = make_layer(x, basic_block, planes * 2, planes * 2, m, expansion=basicblock_expansion)  # layer3_
        if m == 2:
            x_d = make_layer(x, basic_block, planes * 2, planes, 0, expansion=basicblock_expansion)  # layer3_d
        else:
            x_d = make_layer(x, basic_block, planes * 2, planes * 2, 0, expansion=basicblock_expansion)  # layer3_d
        x_d = layers.Activation("relu")(x_d)

        x = make_layer(x, basic_block, planes * 2, planes * 4, n, stride=2, expansion=basicblock_expansion)  # layer3
        x = layers.Activation("relu")(x)

        # P Branch
        compression3 = layers.Conv2D(planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression3
        compression3 = layers.BatchNormalization(momentum=bn_mom)(compression3)

        x_ = PagFM(x_, compression3, planes * 2, planes)  # pag3

        if m == 2:
            diff3 = layers.Conv2D(planes, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
            diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)
        else:
            diff3 = layers.Conv2D(planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff3
            diff3 = layers.BatchNormalization(momentum=bn_mom)(diff3)

        diff3 = tf.image.resize(diff3, size=(height_output, width_output), method='bilinear')
        x_d = x_d + diff3

        if augment:
            temp_p = x_

        layer4 = make_layer(x, basic_block, planes * 4, planes * 8, n, stride=2, expansion=basicblock_expansion)  # layer4
        x = layers.Activation("relu")(layer4)

        x_ = layers.Activation("relu")(x_)
        x_ = make_layer(x_, basic_block, planes * 2, planes * 2, m, expansion=basicblock_expansion)  # layer4_

        x_d = layers.Activation("relu")(x_d)
        if m == 2:
            x_d = make_layer(x_d, bottleneck_block, planes, planes, 1, expansion=bottleneck_expansion)  # layer4_d
        else:
            x_d = make_layer(x_d, basic_block, planes * 2, planes * 2, 0, expansion=basicblock_expansion)  # layer4_d
            x_d = layers.Activation("relu")(x_d)

        compression4 = layers.Conv2D(planes * 2, kernel_size=(1, 1), use_bias=False)(x)  # compression4
        compression4 = layers.BatchNormalization(momentum=bn_mom)(compression4)
        x_ = PagFM(x_, compression4, planes * 2, planes)  # pag4

        diff4 = layers.Conv2D(planes * 2, kernel_size=(3, 3), padding='same', use_bias=False)(x)  # diff4
        diff4 = layers.BatchNormalization(momentum=bn_mom)(diff4)
        diff4 = tf.image.resize(diff4, size=(height_output, width_output), method='bilinear')
        x_d = x_d + diff4

        if augment:
            temp_d = x_d

        x_ = layers.Activation("relu")(x_)
        x_ = make_layer(x_, bottleneck_block, planes * 2, planes * 2, 1, expansion=bottleneck_expansion)  # layer5_

        x_d = layers.Activation("relu")(x_d)
        x_d = make_layer(x_d, bottleneck_block, planes * 2, planes * 2, 1, expansion=bottleneck_expansion)  # layer5_d

        layer5 = make_layer(x, bottleneck_block, planes * 8, planes * 8, 2, stride=2,
                            expansion=bottleneck_expansion)  # layer5
        if m == 2:
            spp = PAPPM(layer5, ppm_planes, planes * 4)  # spp
            x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
            dfm = Light_Bag(x_, x, x_d, planes * 4)  # dfm
        else:
            spp = DAPPPM(layer5, ppm_planes, planes * 4)  # spp
            x = tf.image.resize(spp, size=(height_output, width_output), method='bilinear')
            dfm = Bag(x_, x, x_d, planes * 4)  # dfm

        x_ = segmentation_head(dfm, head_planes, num_classes)  # final_layer

        # Prediction Head
        if augment:
            seghead_p = segmentation_head(temp_p, head_planes, num_classes)
            seghead_d = segmentation_head(temp_d, planes, 1)
            model_output = [seghead_p, x_, seghead_d]
        else:
            model_output = [x_]

        model = models.Model(inputs=[x_in], outputs=model_output)

        # set weight initializers
        for layer in model.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel_initializer = tf.keras.initializers.he_normal()
            if hasattr(layer, 'beta_initializer'):  # for BatchNormalization
                layer.beta_initializer = "zeros"
                layer.gamma_initializer = "ones"

        self.__model = model
        self.__model_name = 'pidnet'

    @property
    def model(self):
        return self.__model

    @property
    def model_name(self):
        return self.__model_name


class PIDNetSmall(PIDNet):
    def __init__(self,
                 input_shape,
                 number_of_classes):
        super().__init__(input_shape=input_shape,
                         num_classes=number_of_classes,
                         m=2,
                         n=3,
                         planes=32,
                         ppm_planes=96,
                         head_planes=128,
                         augment=False)
        self.__model_name = 'pidnet_small'

    @property
    def model_name(self):
        return self.__model_name
