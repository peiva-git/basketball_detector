import unittest

from parameterized import parameterized
from detector.tasks.predict_ball_locations import patch_indexes_from_coordinates


class PredictionTestCase(unittest.TestCase):

    @parameterized.expand([
        # first row
        (0, 0, [0]),
        (0, 10, [0, 1]),
        (0, 20, [0, 1, 2]),
        (0, 30, [0, 1, 2, 3]),
        (0, 40, [0, 1, 2, 3, 4]),
        (0, 50, [1, 2, 3, 4, 5]),
        (0, 60, [2, 3, 4, 5, 6]),
        (0, 980, [94, 95, 96]),
        (0, 990, [95, 96]),
        (0, 1000, [96]),
        # second row
        (10, 0, [0, 97]),
        (10, 10, [0, 1, 97, 98]),
        (10, 20, [0, 1, 2, 97, 98, 99]),
        (10, 30, [0, 1, 2, 3, 97, 98, 99, 100])
    ])
    def test_patch_indexes_from_coordinates(self, row: int, column: int, expected: list[int]):
        self.assertListEqual(patch_indexes_from_coordinates(row, column, 512, 1024, stride=10, window_size=50), expected)



