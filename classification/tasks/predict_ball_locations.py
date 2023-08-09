import cv2 as cv
import tensorflow as tf


def divide_frame_into_patches(frame, stride: int = 5, window_size: int = 50) -> [(int, int, cv.UMat)]:
    # could try out with a stride of 10 and a window_size of 100 as well
    height, width, channels = frame.shape
    position_height = 0
    position_width = 0
    number_of_width_windows = int(width / stride) - int(window_size / stride)
    number_of_height_windows = int(height / stride) - int(window_size / stride)
    # with ~4ms inference time on a single patch, a whole image is evaluated in approx. 5 minutes
    # with a window size of 50 and a stride of 5
    # with a window size of 100 and a stride of 10, an image is evaluated in approx. 1 minute
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


if __name__ == '__main__':
    # capture = cv.VideoCapture('/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019'
    #                           '-20_legabasket/pallacanestro_trieste-virtus_roma/final_cut.mp4')
    # if not capture.isOpened():
    #     print("Can't open video file")
    #     exit()
    # while True:
    #     ret, frame = capture.read()
    #     if not ret:
    #         print("Can't read next frame (stream end?). Exiting...")
    #         break
    #     image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #
    #     if cv.waitKey(1) == ord('q'):
    #         break
    #
    # capture.release()
    # cv.destroyAllWindows()
    image = cv.imread('/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket'
                      '/pallacanestro_trieste-virtus_roma/frame_00092.png')
    image_patches = divide_frame_into_patches(image, stride=5, window_size=50)
    patches_only = [element[2] for element in image_patches]
    patches_dataset = tf.data.Dataset.from_tensor_slices(patches_only)
    patches_dataset = patches_dataset.batch(batch_size=64)
