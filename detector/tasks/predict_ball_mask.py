import os
import pathlib
import random

from vidgear.gears import CamGear

import cv2 as cv
import numpy as np
import fastdeploy as fd

TEST_DATA_DIR = '/mnt/DATA/tesi/dataset/dataset_paddleseg/images/'
MODEL_PATH = '/home/peiva/ppliteSeg/inference_model.onnx'
# '/home/ubuntu/PaddleSeg/output/inference_model.onnx'
batch_size = 5


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


if __name__ == '__main__':
    print('Building model...')
    option = fd.RuntimeOption()
    option.use_gpu()
    # model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    # params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    # config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
    model_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/model.pdmodel')
    params_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/model.pdiparams')
    config_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model_softmax/deploy.yaml')
    model = fd.vision.segmentation.PaddleSegModel(
        str(model_file), str(params_file), str(config_file), runtime_option=option
    )
    # model.postprocessor.store_score_map = True
    model.postprocessor.apply_softmax = True
    print('Reading video...')
    stream = CamGear(
        source='/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket/pallacanestro_trieste-virtus_roma/final_cut.mp4'
        #source='/home/ubuntu/test_video.mp4'
    ).start()
    index = 1
    while True:
        frame = stream.read()
        if frame is None:
            break
        frame_resized = cv.resize(frame, (2048, 1024))
        random_crops = generate_random_crops(frame_resized, number_of_crops=batch_size, variance=100)
        print(f'Predicting frame {index}...')
        results = model.batch_predict([cv.resize(crop[2], (2048, 1024)) for crop in random_crops])
        averaged_heatmap = np.mean([np.reshape(result.label_map, (1024, 2048)) for result in results], axis=0)
        rescaled_heatmap = averaged_heatmap * 255
        # vis_im = fd.vision.vis_segmentation(frame_resized, result, weight=0.5)
        # print(f'Writing overlay {index} to disk...')
        # cv.imwrite(f'/home/ubuntu/results/overlay{index}.png', vis_im)
        cv.imshow('Output', rescaled_heatmap.astype(np.uint8))
        index += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    stream.stop()
