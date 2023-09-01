import unittest

from detector.tasks.predict_ball_locations import patch_indexes_from_coordinates


class PredictionTestCase(unittest.TestCase):

    def test_patch_indexes_from_coordinates(self):
        self.assertListEqual(patch_indexes_from_coordinates(0, 0), [0],
                             'Left upper corner pixel should be embedded only in the first patch')


