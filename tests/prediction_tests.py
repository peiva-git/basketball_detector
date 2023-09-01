import unittest

from detector.tasks.predict_ball_locations import patch_indexes_from_coordinates


class PredictionTestCase(unittest.TestCase):

    def test_patch_indexes_from_coordinates(self):
        self.assertListEqual(patch_indexes_from_coordinates(0, 0, stride=10, window_size=50), [0],
                             'Left upper corner pixel should be embedded only in the first patch')
        self.assertListEqual(patch_indexes_from_coordinates(0, 10, stride=10, window_size=50), [0, 1])
        self.assertListEqual(patch_indexes_from_coordinates(0, 20, stride=10, window_size=50), [0, 1, 2])
        self.assertListEqual(patch_indexes_from_coordinates(0, 30, stride=10, window_size=50), [0, 1, 2, 3])
        self.assertListEqual(patch_indexes_from_coordinates(0, 40, stride=10, window_size=50), [0, 1, 2, 3, 4])


