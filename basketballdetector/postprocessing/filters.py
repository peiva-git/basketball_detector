from skimage.morphology import remove_small_objects

import numpy as np


def remove_connected_areas(mask: np.ndarray, min_size: int, max_size: int) -> np.ndarray:
    small_removed = remove_small_objects(mask.astype(bool), min_size=min_size)
    large_objects_mask = remove_small_objects(small_removed, min_size=max_size)
    return small_removed.astype(np.uint8) - large_objects_mask.astype(np.uint8)
