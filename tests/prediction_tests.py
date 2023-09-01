import unittest

from detector.tasks.predict_ball_locations import patch_indexes_from_coordinates


class PredictionTestCase(unittest.TestCase):

    def test_patch_indexes_from_coordinates(self):
        self.assertListEqual(patch_indexes_from_coordinates(0, 0, stride=20, window_size=112), [0],
                             'Left upper corner pixel should be embedded only in the first patch')
        self.assertListEqual(patch_indexes_from_coordinates(0, 21, stride=20, window_size=112), [0, 1])
        self.assertListEqual(patch_indexes_from_coordinates(0, 40, stride=20, window_size=112), [0, 1, 2])


