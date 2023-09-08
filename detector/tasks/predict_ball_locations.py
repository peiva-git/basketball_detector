import math
import pathlib
import time
from collections import defaultdict
from itertools import product
from statistics import mean

import cv2 as cv
import tensorflow as tf
import numpy as np


class PatchesSequence(tf.keras.utils.Sequence):

    def __init__(self, patches: [], batch_size: int = 64):
        self.__patches = patches
        self.__batch_size = batch_size

    def __getitem__(self, index):
        low = index * self.__batch_size
        high = min(low + self.__batch_size, len(self.__patches))
        patches_batch = self.__patches[low:high]
        return np.array([patch for patch in patches_batch])

    def __len__(self):
        return math.ceil(len(self.__patches) / self.__batch_size)


def divide_frame_into_patches(frame, stride: int = 5, window_size: int = 50) -> [(int, int, cv.UMat)]:
    # could try out with a stride of 10 and a window_size of 100 as well
    # the origin of the coordinates' system is in the upper left corner of the image
    # with the x-axis facing to the right, and the y-axis facing down
    height, width, _ = frame.shape
    position_height = 0
    position_width = 0
    number_of_width_windows = int(width / stride) - int(window_size / stride)
    number_of_height_windows = int(height / stride) - int(window_size / stride)

    patches = []
    for window_height_index in range(number_of_height_windows):
        for window_width_index in range(number_of_width_windows):
            current_patch = frame[
                            position_height:position_height + window_size,
                            position_width:position_width + window_size
                            ]
            current_patch_rgb = cv.cvtColor(current_patch, cv.COLOR_BGR2RGB)
            patches.append((position_height, position_width, current_patch_rgb))
            position_width += stride
        position_width = 0
        position_height += stride

    return patches


def write_frame_patches_to_disk(frame, target_directory: str,
                                stride: int = 5, window_size: int = 50,
                                verbose: bool = True):
    target = pathlib.Path(target_directory)
    count = 1
    image_patches = divide_frame_into_patches(frame, stride, window_size)
    for position_y, position_x, patch in image_patches:
        patch = cv.cvtColor(patch, cv.COLOR_RGB2BGR)
        cv.imwrite(str(target / f'patch_x{position_x}_y{position_y}.png'), patch)
        if verbose:
            print(f'Written image {count} out of {len(image_patches)}')
        count += 1


def obtain_predictions(frame,
                       model_path: str,
                       stride: int = 5,
                       window_size: int = 50) -> ([int, int, cv.UMat], [int, int]):
    model_path = pathlib.Path(model_path)
    patches_with_positions = divide_frame_into_patches(frame, stride=stride, window_size=window_size)
    patches_only = [element[2] for element in patches_with_positions]
    model = tf.keras.models.load_model(str(model_path))
    patches_sequence = PatchesSequence(patches_only)
    predictions = model.predict(patches_sequence)
    return patches_with_positions, predictions


def annotate_frame_with_ball_patches(frame, patches_with_positions, predictions, window_size: int = 50,
                                     threshold: float = 0.9) -> cv.UMat:
    for index, (height_coordinate, width_coordinate, image_patch) in enumerate(patches_with_positions):
        prediction = predictions[index]
        if prediction[0] >= threshold:
            # more likely that the patch is a ball
            cv.rectangle(
                frame,
                (width_coordinate, height_coordinate),
                (width_coordinate + window_size, height_coordinate + window_size),
                color=(0, 255, 0)
            )
        else:
            # more likely that the patch is not a ball
            pass
    return frame


def obtain_heatmap(frame, predictions, patches_with_positions, window_size: int = 50, stride: int = 10):
    frame_height, frame_width, _ = frame.shape
    heatmap = np.zeros((frame_height, frame_width), np.float32)
    number_of_width_windows = int(frame_width / stride) - int(window_size / stride)
    number_of_height_windows = int(frame_height / stride) - int(window_size / stride)
    patch_indexes_by_pixel = defaultdict(set)
    print('Building pixel -> indexes dictionary...')
    map_pixels_to_patch_indexes(patch_indexes_by_pixel, patches_with_positions, window_size)
    for row, column in product(range(frame_height), range(frame_width)):
        pixel_indexes = patch_indexes_by_pixel[(row, column)]
            # __get_indexes(row, column,
            #                           number_of_height_windows, number_of_width_windows,
            #                           window_size, stride)
        print(f'Found indexes for pixel ({row},{column})')
        if len(pixel_indexes) != 0:
            patches_ball_probabilities = [predictions[patch_index][0] for patch_index in pixel_indexes]
            pixel_ball_probability = sum(patches_ball_probabilities) / len(pixel_indexes)
            heatmap[row, column] = pixel_ball_probability
    heatmap_rescaled = heatmap * 255
    return heatmap_rescaled.astype(np.uint8, copy=False)


def patch_indexes_from_coordinates(row: int, column: int,
                                   frame_height: int, frame_width: int,
                                   window_size: int = 50, stride: int = 10) -> list[int]:
    number_of_width_windows = int(frame_width / stride) - int(window_size / stride)
    number_of_height_windows = int(frame_height / stride) - int(window_size / stride)
    return __get_indexes(row, column, number_of_height_windows, number_of_width_windows, window_size, stride)


def __get_indexes(row: int, column: int,
                  number_of_height_windows: int, number_of_width_windows: int,
                  window_size: int = 50, stride: int = 10) -> list[int]:
    # comments assuming default values
    # column < 40 (less than 5 indexes per row)
    if column < stride * (int(window_size / stride) - 1):
        # rows < 50
        if row < stride * (int(window_size / stride)):
            result = []
            for mult in range(int(row / stride) + 1):
                result.extend([i for i in range(number_of_width_windows * mult,
                                                number_of_width_windows * mult + int(column / stride) + 1)])
            return result
        # 50 <= row < 460
        if stride * (int(window_size / stride)) \
                <= row < stride * number_of_height_windows:
            result = []
            for mult in range(int((row - window_size) / stride) + 1, int(row / stride) + 1):
                result.extend([
                    i for i in
                    range(number_of_width_windows * mult + int(column / window_size),
                          number_of_width_windows * mult + int(column / stride) + 1)
                ])
            return result
        # 460 <= row < 510
        if stride * number_of_height_windows \
                <= row < stride * number_of_height_windows + window_size:
            result = []
            for mult in range(int((row - window_size) / stride), number_of_height_windows):
                result.extend([
                    i for i in
                    range(number_of_width_windows * mult + int(column / window_size),
                          number_of_width_windows * mult + int(column / stride) + 1)
                ])
            return result
        else:
            return []

    # 40 <= column < 970
    if stride * (int(window_size / stride) - 1) \
            <= column < stride * (number_of_width_windows - int(window_size / stride)) + window_size:
        # row < 50
        if row < stride * (int(window_size / stride)):
            result = []
            for mult in range(int(row / stride) + 1):
                result.extend([
                    i for i in
                    range(int(column / stride) - int(window_size / stride) + 1 + number_of_width_windows * mult,
                          int(column / stride) + 1 + number_of_width_windows * mult)
                ])
            return result
        # 50 <= row < 460
        if stride * (int(window_size / stride)) \
                <= row < stride * (number_of_height_windows - int(window_size / stride)) + window_size:
            result = []
            for mult in range(int((row - window_size) / stride) + 1, int(row / stride) + 1):
                result.extend([i for i in range(number_of_width_windows * mult + int(column / window_size),
                                                number_of_width_windows * mult + int(column / stride) + 1)])
            return result
        # 460 <= row < 510
        if stride * (number_of_height_windows - int(window_size / stride)) + window_size \
                <= row < stride * number_of_height_windows + window_size:
            result = []
            for mult in range(int((row - window_size) / stride), number_of_height_windows):
                result.extend([i for i in range(number_of_width_windows * mult + int(column / window_size),
                                                number_of_width_windows * mult + int(column / stride) + 1)])
            return result
        else:
            return []

    # 970 <= column < 1020
    if stride * (number_of_width_windows - int(window_size / stride)) + window_size \
            <= column < stride * number_of_width_windows + window_size:
        # row < 50
        if row < stride * (int(window_size / stride)):
            result = []
            for mult in range(int(row / stride) + 1):
                result.extend(sorted([
                    i for i in
                    range(number_of_width_windows * (mult + 1) - 1,
                          int(column / stride) - int(window_size / stride) + number_of_width_windows * mult - 1,
                          -1)
                ]))
            return result
        # 50 <= row < 460
        if stride * (int(window_size / stride)) \
                <= row < stride * (number_of_height_windows - int(window_size / stride)) + window_size:
            result = []
            for mult in range(int((row - window_size) / stride) + 1, int(row / stride) + 1):
                result.extend(sorted([
                    i for i in
                    range(number_of_width_windows * (mult + 1) - 1,
                          int(column / stride) - int(window_size / stride) + number_of_width_windows * mult - 1,
                          -1)
                ]))
            return result
        # 460 <= row < 510
        if stride * (number_of_height_windows - int(window_size / stride)) + window_size \
                <= row < stride * number_of_height_windows + window_size:
            result = []
            for mult in range(int((row - window_size) / stride), number_of_height_windows):
                result.extend(sorted([
                    i for i in
                    range(number_of_width_windows * (mult + 1) - 1,
                          int(column / stride) - int(window_size / stride) + number_of_width_windows * mult - 1,
                          -1)
                ]))
            return result
        else:
            return []
    else:
        return []


def map_pixels_to_patch_indexes(patch_indexes_by_pixel, patches_with_positions, window_size: int):
    for index, (patch_position_y, patch_position_x, _) in enumerate(patches_with_positions):
        iterate_over_patch(index, patch_indexes_by_pixel, patch_position_x, patch_position_y, window_size)


def iterate_over_patch(index, patch_indexes_by_pixel, patch_position_x, patch_position_y, window_size):
    for row, column in product(range(patch_position_y, patch_position_y + window_size),
                               range(patch_position_x, patch_position_x + window_size)):
        patch_indexes_by_pixel[(row, column)].add(index)


def find_max_pixel(heatmap) -> (int, int):
    max_index = heatmap.argmax()
    _, heatmap_width = heatmap.shape
    return max_index - int(max_index / heatmap_width) * heatmap_width, int(max_index / heatmap_width)


def annotate_frame(frame,
                   heatmap,
                   threshold_delta: int = 10,
                   margin: int = 0) -> ((int, int, int, int), cv.UMat):
    max_pixel = find_max_pixel(heatmap)
    heatmap_height, heatmap_width = heatmap.shape
    mask = np.zeros((heatmap_height + 2, heatmap_width + 2), np.uint8)
    _, _, _, bounding_box = cv.floodFill(
        image=heatmap, mask=mask, seedPoint=max_pixel, newVal=255, loDiff=threshold_delta,
        flags=8 | (255 << 8) | cv.FLOODFILL_FIXED_RANGE | cv.FLOODFILL_MASK_ONLY
    )
    cv.rectangle(
        frame,
        (bounding_box[0] - margin, bounding_box[1] - margin),
        (bounding_box[0] + bounding_box[2] + margin, bounding_box[1] + bounding_box[3] + margin),
        color=(0, 255, 0)
    )
    return bounding_box, mask


def write_detections_video(input_video_path: str,
                           target_video_path: str,
                           model_path: str):
    input_path = pathlib.Path(input_video_path)
    target_path = pathlib.Path(target_video_path)
    model_path = pathlib.Path(model_path)
    capture = cv.VideoCapture(str(input_path))
    out = cv.VideoWriter(str(target_path), fourcc=0, fps=0, frameSize=(1920, 1080))
    if not capture.isOpened():
        print("Can't open video file")
        return
    counter = 1
    frame_processing_times = []
    while counter <= 100:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Processing frame {counter} out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        start = time.time()
        _, patches_predictions = obtain_predictions(
            image, str(model_path), stride=10, window_size=50
        )
        heatmap = obtain_heatmap(image, patches_predictions, window_size=50, stride=10)
        annotate_frame(image, heatmap)
        out.write(image)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}'
              f' out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(frame_processing_times)} seconds')
        counter += 1
        # cv.imshow(f'frame {counter}', image)
        # if cv.waitKey(1) == ord('q'):
        #     break
    capture.release()
    out.release()
    cv.destroyAllWindows()


def write_image_sequence_from_video(input_video_path: str,
                                    target_directory_path: str,
                                    model_path: str):
    input_path = pathlib.Path(input_video_path)
    target_path = pathlib.Path(target_directory_path)
    model_path = pathlib.Path(model_path)
    capture = cv.VideoCapture(str(input_path))
    if not capture.isOpened():
        print("Can't open video file")
        return
    counter = 1
    frame_processing_times = []
    while counter <= 10:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Processing frame {counter} out of {10}')
        start = time.time()
        print('Obtaining predictions...')
        patches_with_positions, patches_predictions = obtain_predictions(
            image, str(model_path), window_size=50, stride=10
        )
        print('Building heatmap from predictions...')
        heatmap = obtain_heatmap(image, patches_predictions, patches_with_positions, window_size=50, stride=10)
        _, mask = annotate_frame(image, heatmap)
        cv.imwrite(str(target_path / f'frame_{counter}.png'), image)
        cv.imwrite(str(target_path / f'heatmap_{counter}.png'), heatmap)
        cv.imwrite(str(target_path / f'mask_{counter}.png'), mask)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}'
              f' out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(frame_processing_times)} seconds')
        counter += 1
        # cv.imshow(f'frame {counter}', image)
        # if cv.waitKey(1) == ord('q'):
        #     break
    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    # with ~4ms inference time on a single patch, a whole image is evaluated in approx. 5 minutes
    # with a window size of 50 and a stride of 5
    # with a window size of 100 and a stride of 10, an image is evaluated in approx. 1 minute
    # these values are estimated based on the mobilenetv2 inference time measurements displayed here
    # https://keras.io/api/applications/#available-models
    write_image_sequence_from_video(input_video_path='/home/peiva/experiments/test_videos/final_cut.mp4',
                                    target_directory_path='/home/peiva/experiments/',
                                    model_path='/home/peiva/mobilenet/first_test/models/Keras_v3/mobilenetv2.keras')
    # write_image_sequence_from_video(input_video_path='/home/ubuntu/test_videos/final_cut.mp4',
    #                                 target_directory_path='/home/ubuntu/test_videos',
    #                                 model_path='/home/ubuntu/basketball_detector/out/models/Keras_v3/mobilenetv2.keras')
    # write_detections_video(input_video_path='/home/ubuntu/test_videos/final_cut.mp4',
    #                        target_video_path='home/ubuntu/test_videos/annotated.mp4',
    #                        model_path='/home/ubuntu/basketball_detector/out/models/Keras_v3/mobilenetv2.keras')
