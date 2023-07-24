import unittest

from classification import Classifier


class ModelsTestCase(unittest.TestCase):
    __SIMPLE_CLASSIFIER = Classifier()
    __RESNET_CLASSIFIER = Classifier(model_name='resnet-classifier')

    def test_model_name(self):
        self.assertIn(self.__SIMPLE_CLASSIFIER.model_name, ['simple-classifier', 'resnet-classifier'],
                      'incorrect model name')
        self.assertIn(self.__RESNET_CLASSIFIER.model_name, ['simple-classifier', 'resnet-classifier'],
                      'incorrect model name')
