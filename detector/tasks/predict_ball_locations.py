import pathlib
import time
from itertools import product
from statistics import mean

import cv2 as cv
import numpy
import tensorflow as tf
import numpy as np


def divide_frame_into_patches(frame, stride: int = 5, window_size: int = 50) -> [(int, int, cv.UMat)]:
    # could try out with a stride of 10 and a window_size of 100 as well
    # the origin of the coordinates' system is in the upper left corner of the image
    # with the x-axis facing to the right, and the y-axis facing down
    height, width, channels = frame.shape
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
    print('Organizing patches into a tensorflow dataset...')
    patches_dataset = tf.data.Dataset.from_tensor_slices(patches_only)
    patches_dataset = patches_dataset.batch(batch_size=64)
    patches_dataset = patches_dataset.prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model(str(model_path))
    predictions = model.predict(patches_dataset, callbacks=[tf.keras.callbacks.ProgbarLogger()])
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


def obtain_heatmap(frame, patches_with_positions, predictions, window_size: int = 50):
    frame_height, frame_width, _ = frame.shape
    heatmap = np.zeros((frame_height, frame_width), numpy.float32)
    patch_indexes_by_pixel = dict()

    for index, (patch_position_y, patch_position_x, patch) in enumerate(patches_with_positions):
        for row, column in product(range(patch_position_y, patch_position_y + window_size),
                                   range(patch_position_x, patch_position_x + window_size)):
            try:
                patch_indexes_by_pixel[(row, column)].append(index)
            except KeyError:
                patch_indexes_by_pixel[(row, column)] = [index]
    for row, column in product(range(frame_height), range(frame_width)):
        try:
            patches_ball_probabilities = \
                [predictions[patch_index][0] for patch_index in patch_indexes_by_pixel[(row, column)]]
            pixel_ball_probability = sum(patches_ball_probabilities) / len(patch_indexes_by_pixel[(row, column)])
            heatmap[row, column] = pixel_ball_probability
        except KeyError:
            print(f'No patch contains pixel x: {column}, y: {row}, assigning a probability of 0')
            pass
    heatmap_rescaled = heatmap * 255
    return heatmap_rescaled.astype(np.uint8, copy=False)


def find_max_pixel(heatmap) -> (int, int):
    max_index = heatmap.argmax()
    _, heatmap_width = heatmap.shape
    return max_index - int(max_index / heatmap_width) * heatmap_width, int(max_index / heatmap_width)


def annotate_frame(frame, heatmap, threshold: int = 10, margin: int = 10) -> (int, int, int, int):
    max_pixel = find_max_pixel(heatmap)
    _, _, _, bounding_box = cv.floodFill(heatmap, None, seedPoint=max_pixel, newVal=255, loDiff=threshold, flags=8)
    cv.rectangle(
        frame,
        (bounding_box[0] - margin, bounding_box[1] - margin),
        (bounding_box[0] + bounding_box[2] + margin, bounding_box[1] + bounding_box[3] + margin),
        color=(0, 255, 0)
    )
    return bounding_box


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
    while True:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Processing frame {counter} out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        start = time.time()
        patches_and_positions, patches_predictions = obtain_predictions(
            image, str(model_path)
        )
        heatmap = obtain_heatmap(image, patches_and_positions, patches_predictions)
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


if __name__ == '__main__':
    # with ~4ms inference time on a single patch, a whole image is evaluated in approx. 5 minutes
    # with a window size of 50 and a stride of 5
    # with a window size of 100 and a stride of 10, an image is evaluated in approx. 1 minute
    # these values are estimated based on the mobilenetv2 inference time measurements displayed here
    # https://keras.io/api/applications/#available-models
    write_detections_video(input_video_path='/home/ubuntu/test_videos/final_cut.mp4',
                           target_video_path='home/ubuntu/test_videos/annotated.mp4',
                           model_path='/home/ubuntu/basketball_detector/out/models/Keras_v3/mobilenetv2.keras')
