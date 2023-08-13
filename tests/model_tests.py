import unittest

from detector.models.classification import SimpleClassifier


class ModelsTestCase(unittest.TestCase):
    __SIMPLE_CLASSIFIER = SimpleClassifier()
    __RESNET_CLASSIFIER = SimpleClassifier(model_name='resnet-classifier')

    def test_model_name(self):
        self.assertIn(self.__SIMPLE_CLASSIFIER.model_name, ['simple-classifier', 'resnet-classifier'],
                      'incorrect model name')
        self.assertIn(self.__RESNET_CLASSIFIER.model_name, ['simple-classifier', 'resnet-classifier'],
                      'incorrect model name')
