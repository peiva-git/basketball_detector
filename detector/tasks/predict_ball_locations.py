import pathlib
import time

import cv2 as cv
import tensorflow as tf


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
                       stride: int = 5,
                       window_size: int = 50) -> ([int, int, cv.UMat], [int, int]):
    print('Frame read, dividing into patches...')
    patches_with_positions = divide_frame_into_patches(frame, stride=stride, window_size=window_size)
    patches_only = [element[2] for element in patches_with_positions]
    print('Organizing patches into a tensorflow dataset...')
    patches_dataset = tf.data.Dataset.from_tensor_slices(patches_only)
    patches_dataset = patches_dataset.batch(batch_size=64)
    patches_dataset = patches_dataset.prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model('/home/peiva/mobilenet/models/Keras_v3/mobilenetv2.keras')
    predictions = model.predict(patches_dataset, callbacks=[tf.keras.callbacks.ProgbarLogger()])
    return patches_with_positions, predictions


def annotate_frame(frame, patches_with_positions, predictions, window_size: int = 50,
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


def obtain_heatmap(frame, patches_with_positions, predictions):

    pass


def write_detections_video(input_video_path: str,
                           target_video_path: str):
    input_path = pathlib.Path(input_video_path)
    target_path = pathlib.Path(target_video_path)
    capture = cv.VideoCapture(str(input_path))
    out = cv.VideoWriter(str(target_path), fourcc=0, fps=0)
    if not capture.isOpened():
        print("Can't open video file")
        exit()
    counter = 1
    while True:
        ret, image = capture.read()
        if not ret:
            print("Can't read next frame (stream end?). Exiting...")
            break
        print(f'Annotating frame {counter} out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
        start = time.time()
        patches_and_positions, patches_predictions = obtain_predictions(image)
        annotate_frame(image, patches_and_positions, patches_predictions)
        out.write(image)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}'
              f' out of {int(capture.get(cv.CAP_PROP_FRAME_COUNT))}')
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
    write_detections_video('/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste'
                           '/stagione_2019'
                           '-20_legabasket/pallacanestro_trieste-virtus_roma/final_cut.mp4',
                           '/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste'
                           '/stagione_2019-20_legabasket/'
                           'pallacanestro_trieste-virtus_roma/annotated.mp4')
