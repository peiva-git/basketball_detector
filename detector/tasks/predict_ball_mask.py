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


def generate_random_crops(image, number_of_crops: int) -> []:
    image_height, image_width, _ = image.shape
    crops = []
    for i in range(number_of_crops):
        crop_row_start = random.randint(0, )
    [
        image[crop_row_start:crop_row_start + height, crop_column_start:crop_column_start + width]
        for crop_row_start, height, crop_column_start, width
        in
    ]
    return crops


if __name__ == '__main__':
    print('Building model...')
    option = fd.RuntimeOption()
    option.use_gpu()
    # model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    # params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    # config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
    model_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdmodel')
    params_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdiparams')
    config_file = pathlib.Path('/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/deploy.yaml')
    model = fd.vision.segmentation.PaddleSegModel(
        str(model_file), str(params_file), str(config_file), runtime_option=option
    )
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
        random_crops = generate_random_crops(frame_resized, number_of_crops=batch_size)
        print(f'Predicting frame {index}...')
        results = model.batch_predict(random_crops)
        averaged_heatmap = np.mean([np.reshape(result.label_map, (1024, 2048)) for result in results], axis=0)
        vis_im = fd.vision.vis_segmentation(frame_resized, result, weight=0.5)
        # print(f'Writing overlay {index} to disk...')
        # cv.imwrite(f'/home/ubuntu/results/overlay{index}.png', vis_im)
        cv.imshow('Output', vis_im)
        index += 1

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    stream.stop()
