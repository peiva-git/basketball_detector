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
        (10, 30, [0, 1, 2, 3, 97, 98, 99, 100]),
        (10, 40, [0, 1, 2, 3, 4, 97, 98, 99, 100, 101]),
        (10, 50, [1, 2, 3, 4, 5, 98, 99, 100, 101, 102]),
        (10, 980, [94, 95, 96, 191, 192, 193]),
        (10, 990, [95, 96, 192, 193]),
        (10, 1000, [96, 193]),
        # third row
        (20, 0, [0, 97, 194]),
        (20, 10, [0, 1, 97, 98, 194, 195]),
        (20, 20, [0, 1, 2, 97, 98, 99, 194, 195, 196]),
        (20, 30, [0, 1, 2, 3, 97, 98, 99, 100, 194, 195, 196, 197]),
        (20, 40, [0, 1, 2, 3, 4, 97, 98, 99, 100, 101, 194, 195, 196, 197, 198]),
        (20, 980, [94, 95, 96, 191, 192, 193, 288, 289, 290]),
        (20, 990, [95, 96, 192, 193, 289, 290]),
        (20, 1000, [96, 193, 290]),
        # fourth row
        (30, 0, [0, 97, 194, 291]),
        (30, 10, [0, 1, 97, 98, 194, 195, 291, 292]),
        (30, 20, [0, 1, 2, 97, 98, 99, 194, 195, 196, 291, 292, 293]),
        (30, 30, [0, 1, 2, 3, 97, 98, 99, 100, 194, 195, 196, 197, 291, 292, 293, 294]),
        (30, 40, [0, 1, 2, 3, 4, 97, 98, 99, 100, 101, 194, 195, 196, 197, 198, 291, 292, 293, 294, 295]),
        (30, 50, [1, 2, 3, 4, 5, 98, 99, 100, 101, 102, 195, 196, 197, 198, 199, 292, 293, 294, 295, 296]),
        (30, 980, [94, 95, 96, 191, 192, 193, 288, 289, 290, 385, 386, 387]),
        (30, 990, [95, 96, 192, 193, 289, 290, 386, 387]),
        (30, 1000, [96, 193, 290, 387]),
        # fifth row
        (40, 0, [0, 97, 194, 291, 388]),
        (40, 10, [0, 1, 97, 98, 194, 195, 291, 292, 388, 389]),
        (40, 20, [0, 1, 2, 97, 98, 99, 194, 195, 196, 291, 292, 293, 388, 389, 390]),
        (40, 30, [0, 1, 2, 3, 97, 98, 99, 100, 194, 195, 196, 197, 291, 292, 293, 294, 388, 389, 390, 391]),
        (40, 40, [0, 1, 2, 3, 4, 97, 98, 99, 100, 101, 194, 195, 196, 197, 198, 291, 292, 293, 294, 295, 388, 389, 390, 391, 392]),
        (40, 980, [94, 95, 96, 191, 192, 193, 288, 289, 290, 385, 386, 387, 482, 483, 484]),
        (40, 990, [95, 96, 192, 193, 289, 290, 386, 387, 483, 484]),
        (40, 1000, [96, 193, 290, 387, 484]),
        # sixth row
        (50, 0, [97, 194, 291, 388, 485]),
        (50, 10, [97, 98, 194, 195, 291, 292, 388, 389, 485, 486]),
        (50, 20, [97, 98, 99, 194, 195, 196, 291, 292, 293, 388, 389, 390, 485, 486, 487]),
        (50, 30, [97, 98, 99, 100, 194, 195, 196, 197, 291, 292, 293, 294, 388, 389, 390, 391, 485, 486, 487, 488]),
        (50, 40, [97, 98, 99, 100, 101, 194, 195, 196, 197, 198, 291, 292, 293, 294, 295, 388, 389, 390, 391, 392, 485, 486, 487, 488, 489]),
        (50, 980, [191, 192, 193, 288, 289, 290, 385, 386, 387, 482, 483, 484, 579, 580, 581]),
        (50, 990, [192, 193, 289, 290, 386, 387, 483, 484, 580, 581]),
        (50, 1000, [193, 290, 387, 484, 581])
    ])
    def test_patch_indexes_from_coordinates(self, row: int, column: int, expected: list[int]):
        self.assertListEqual(patch_indexes_from_coordinates(row, column, 512, 1024, stride=10, window_size=50), expected)



