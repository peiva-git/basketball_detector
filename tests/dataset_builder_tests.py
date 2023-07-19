import unittest
import numpy as np

from classification import DatasetBuilder


class DatasetBuilderTestCase(unittest.TestCase):
    NUMBER_OF_SAMPLE_IMAGES = 18
    builder = DatasetBuilder('../assets/test-sample-data', validation_percentage=0.5)

    def test_image_count(self):
        self.assertEqual(self.builder.image_count, self.NUMBER_OF_SAMPLE_IMAGES, 'incorrect number of images detected')

    def test_class_names(self):
        self.assertTrue(np.all(self.builder.class_names == ['ball', 'no_ball']), 'invalid object classes detected')

    def test_validation_percentage(self):
        self.assertEqual(len(self.builder.train_dataset), self.NUMBER_OF_SAMPLE_IMAGES / 2,
                         'train dataset was not split correctly')
        self.assertEqual(len(self.builder.validation_dataset), self.NUMBER_OF_SAMPLE_IMAGES / 2,
                         'validation dataset was not split correctly')

    def test_dataset_labels(self):
        self.assertIn(self.builder.train_dataset.take(1).get_single_element()[1].numpy(), [0, 1],
                      'invalid class label in train dataset')
        self.assertIn(self.builder.validation_dataset.take(1).get_single_element()[1].numpy(), [0, 1],
                      'invalid class label in validation dataset')
