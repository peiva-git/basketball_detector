import glob
import pathlib

import cv2 as cv


def convert_dataset_to_paddleseg_format(dataset_path: str, target_path: str):
    source = pathlib.Path(dataset_path)
    target = pathlib.Path(target_path)
    images = []
    labels = []

    for match_directory_path in glob.iglob(str(source / '*/*')):
        match_directory = pathlib.Path(match_directory_path)
        match_image_paths = [
            match_image_path
            for match_image_path in glob.iglob(str(match_directory / 'frames/*.png'))
        ]
        match_mask_paths = [
            match_mask_path
            for match_mask_path in glob.iglob(str(match_directory / 'masks/*.png'))
        ]
        match_image_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
        match_mask_paths.sort(key=lambda file_path: int(file_path.split('_')[-1].split('.')[-2]))
        images.extend(match_image_paths)
        labels.extend(match_mask_paths)

    for sample_index in range(len(images)):
        image = cv.imread(images[sample_index])
        label = cv.imread(labels[sample_index])
        resized_image = cv.resize(image, (1024, 512))
        resized_label = cv.resize(label, (1024, 512))
        cv.imwrite(str(target / f'images/image{sample_index + 1}.png'), resized_image)
        cv.imwrite(str(target / f'labels/label{sample_index + 1}.png'), resized_label)
