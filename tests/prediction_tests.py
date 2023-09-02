import unittest

from detector.tasks.predict_ball_locations import patch_indexes_from_coordinates


class PredictionTestCase(unittest.TestCase):

    def test_patch_indexes_from_coordinates(self):
        self.assertListEqual(patch_indexes_from_coordinates(0, 0, 512, 1024, stride=10, window_size=50), [0],
                             'Left upper corner pixel should be embedded only in the first patch')
        self.assertListEqual(patch_indexes_from_coordinates(0, 10, 512, 1024, stride=10, window_size=50), [0, 1])
        self.assertListEqual(patch_indexes_from_coordinates(0, 20, 512, 1024, stride=10, window_size=50), [0, 1, 2])
        self.assertListEqual(patch_indexes_from_coordinates(0, 30, 512, 1024, stride=10, window_size=50), [0, 1, 2, 3])
        self.assertListEqual(patch_indexes_from_coordinates(0, 40, 512, 1024, stride=10, window_size=50), [0, 1, 2, 3, 4])
        self.assertListEqual(patch_indexes_from_coordinates(0, 50, 512, 1024, stride=10, window_size=50), [1, 2, 3, 4, 5])
        self.assertListEqual(patch_indexes_from_coordinates(0, 60, 512, 1024, stride=10, window_size=50), [2, 3, 4, 5, 6])
        self.assertListEqual(patch_indexes_from_coordinates(0, 1000, 512, 1024, stride=10, window_size=50), [96])



