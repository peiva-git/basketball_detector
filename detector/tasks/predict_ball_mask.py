import pathlib
import random
import time

from vidgear.gears import CamGear
from statistics import mean

import cv2 as cv
import numpy as np
import fastdeploy as fd

TEST_DATA_DIR = '/mnt/DATA/tesi/dataset/dataset_paddleseg/images/'
MODEL_PATH = '/home/peiva/ppliteSeg/inference_model.onnx'
# '/home/ubuntu/PaddleSeg/output/inference_model.onnx'
batch_size = 5
input_height = 1024
input_width = 2048
multi_heatmap_prediction = False


def generate_random_crops(image: np.ndarray, number_of_crops: int, variance: int) -> [int, int, np.ndarray]:
    image_height, image_width, _ = image.shape
    crops = []
    crop_width = int(image_width * 0.9)
    crop_height = int(image_height * 0.9)
    first_crop_x, first_crop_y, first_crop = get_random_crop(image, crop_height, crop_width)
    crops.append((first_crop_x, first_crop_y, first_crop))
    for i in range(1, number_of_crops):
        x = random.randint(first_crop_x - variance, first_crop_x + variance)
        y = random.randint(first_crop_y - variance, first_crop_y + variance)
        crop = image[
            max(0, y):min(y + crop_height, image_height),
            max(0, x):min(x + crop_width, image_width)
        ]
        crops.append((max(0, x), max(0, y), crop))
    return crops


def get_random_crop(image: np.ndarray, crop_height: int, crop_width: int) -> (int, int, np.ndarray):
    image_height, image_width, _ = image.shape
    max_x = image_width - crop_width
    max_y = image_height - crop_height
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    return x, y, image[y:y + crop_height, x:x + crop_width]


def draw_crop_boundaries_on_frame(image: np.ndarray, crops: list[(int, int, np.ndarray)]):
    for i, (crop_x, crop_y, crop) in enumerate(crops):
        if i == 0:
            cv.rectangle(
                image,
                (crop_x, crop_y),
                (crop_x + crop.shape[1], crop_y + crop.shape[0]),
                color=(0, 0, 255),
                thickness=5
            )
        else:
            cv.rectangle(
                image,
                (crop_x, crop_y),
                (crop_x + crop.shape[1], crop_y + crop.shape[0]),
                color=(0, 255, 0),
                thickness=5
            )


def get_prediction_with_single_heatmap(
        image: np.ndarray,
        paddle_seg_model: fd.vision.segmentation.PaddleSegModel) -> np.ndarray:
    result = paddle_seg_model.predict(image)
    segmented_image = fd.vision.vis_segmentation(image, result, weight=0.5)
    return segmented_image


def get_prediction_with_multiple_heatmaps(
        image: np.ndarray,
        paddle_seg_model: fd.vision.segmentation.PaddleSegModel,
        number_of_crops: int,
        variance: int):
    random_crops = generate_random_crops(image, number_of_crops, variance)
    results = paddle_seg_model.batch_predict([
        np.pad(crop[2], (
            (crop[1], image.shape[0] - (crop[1] + crop[2].shape[0])),
            (crop[0], image.shape[1] - (crop[0] + crop[2].shape[1])),
            (0, 0)
        ))
        for crop in random_crops
    ])
    averaged_heatmap = np.mean([np.reshape(result.label_map, result.shape) for result in results], axis=0)
    rescaled_heatmap = averaged_heatmap * 255
    return rescaled_heatmap.astype(np.uint8)


if __name__ == '__main__':
    print('Building model...')
    option = fd.RuntimeOption()
    option.use_gpu()
    # model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    # params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    # config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
    if multi_heatmap_prediction:
        model_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/model.pdmodel')
        params_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/model.pdiparams')
        config_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/deploy.yaml')
    else:
        model_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdmodel')
        params_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdiparams')
        config_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/deploy.yaml')
    model = fd.vision.segmentation.PaddleSegModel(
        str(model_file), str(params_file), str(config_file), runtime_option=option
    )
    # model.postprocessor.store_score_map = True
    if multi_heatmap_prediction:
        model.postprocessor.apply_softmax = True
    print('Reading video...')
    stream = CamGear(
        source='/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket/pallacanestro_trieste-virtus_roma/final_cut.mp4'
        #source='/home/ubuntu/test_video.mp4'
    ).start()
    counter = 1
    frame_processing_times = []
    while True:
        frame = stream.read()
        if frame is None:
            break
        frame_resized = cv.resize(frame, (input_width, input_height))
        print(f'Predicting frame {counter}...')
        start = time.time()
        if multi_heatmap_prediction:
            output = get_prediction_with_multiple_heatmaps(frame_resized, model, batch_size, 100)
        else:
            output = get_prediction_with_single_heatmap(frame_resized, model)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {counter}')
        frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(frame_processing_times)} seconds')
        print(f'{1 / (end - start)} FPS')
        cv.imshow('Output', output)
        counter += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    stream.stop()
