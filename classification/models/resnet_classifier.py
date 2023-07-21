import keras_cv


class ResNetClassifier:
    def __init__(self):
        self.model = keras_cv.models.ImageClassifier(
            backbone=keras_cv.models.ResNet152V2Backbone(),
            num_classes=2
        )
