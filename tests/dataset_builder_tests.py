import unittest
import numpy as np

from classification.dataset_builders import ClassificationDatasetBuilder, SegmentationDatasetBuilder


class ClassificationDatasetBuilderTestCase(unittest.TestCase):
    __NUMBER_OF_SAMPLE_IMAGES = 18
    __builder = ClassificationDatasetBuilder('../assets/test-sample-data-classification',
                                             validation_percentage=0.5)

    def test_image_count(self):
        self.assertEqual(self.__builder.number_of_images, self.__NUMBER_OF_SAMPLE_IMAGES,
                         'incorrect number of images detected')

    def test_class_names(self):
        self.assertTrue(np.all(self.__builder.class_names == ['ball', 'no_ball']), 'invalid object classes detected')

    def test_validation_percentage(self):
        self.assertEqual(len(self.__builder.train_dataset), self.__NUMBER_OF_SAMPLE_IMAGES / 2,
                         'train dataset was not split correctly')
        self.assertEqual(len(self.__builder.validation_dataset), self.__NUMBER_OF_SAMPLE_IMAGES / 2,
                         'validation dataset was not split correctly')

    def test_dataset_labels(self):
        self.assertIn(self.__builder.train_dataset.take(1).get_single_element()[1].numpy(), [0, 1],
                      'invalid class label in train dataset')
        self.assertIn(self.__builder.validation_dataset.take(1).get_single_element()[1].numpy(), [0, 1],
                      'invalid class label in validation dataset')


class SegmentationDatasetBuilderTestCase(unittest.TestCase):
    __NUMBER_OF_SAMPLES = 36
    __BUILDER = SegmentationDatasetBuilder('../assets/test-sample-data-segmentation/', validation_percentage=0.5)

    def test_samples_count(self):
        self.assertEqual(self.__BUILDER.number_of_samples, self.__NUMBER_OF_SAMPLES,
                         'incorrect number of samples detected')

    def test_validation_percentage(self):
        self.assertEqual(self.__BUILDER.train_dataset.cardinality().numpy(), self.__NUMBER_OF_SAMPLES / 2,
                         'train dataset was not split correctly')
        self.assertEqual(self.__BUILDER.validation_dataset.cardinality().numpy(), self.__NUMBER_OF_SAMPLES / 2,
                         'validation dataset was not split correctly')

