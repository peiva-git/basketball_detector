import pathlib
import time

from vidgear.gears import CamGear
from statistics import mean

import cv2 as cv
import numpy as np
import fastdeploy as fd

from basketballdetector.predicting.utils import generate_random_crops


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
    # results = [
    #     paddle_seg_model.predict(
    #         np.pad(crop[2], (
    #             (crop[1], image.shape[0] - (crop[1] + crop[2].shape[0])),
    #             (crop[0], image.shape[1] - (crop[0] + crop[2].shape[1])),
    #             (0, 0)
    #         ))
    #     )
    #     for crop in random_crops
    # ]
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


def write_predictions_video():
    pass


def write_image_sequence_predictions():
    pass


def show_prediction_frames(model_file_path: str,
                           params_file_path: str,
                           config_file_path: str,
                           input_video_path: str,
                           stack_heatmaps: bool = False,
                           number_of_crops: int = None,
                           use_trt: bool = False):

    model, video_input = __setup_model(config_file_path, input_video_path, model_file_path, params_file_path,
                                       stack_heatmaps, use_trt)

    print('Reading video...')
    stream = CamGear(source=str(video_input)).start()
    counter = 1
    frame_processing_times = []
    while True:
        frame = stream.read()
        if frame is None:
            break
        frame_resized = cv.resize(frame, (2048, 1024))
        print(f'Predicting frame {counter}...')
        start = time.time()
        if stack_heatmaps:
            output = get_prediction_with_multiple_heatmaps(frame_resized, model, number_of_crops, 100)
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


def __setup_model(config_file_path, input_video_path, model_file_path, params_file_path, stack_heatmaps, use_trt):
    print('Building model...')
    option = fd.RuntimeOption()
    option.use_gpu()
    if use_trt:
        option.use_trt_backend()
        option.set_trt_input_shape('x', [1, 3, 1024, 2048])
    model_file = pathlib.Path(model_file_path)
    params_file = pathlib.Path(params_file_path)
    config_file = pathlib.Path(config_file_path)
    video_input = pathlib.Path(input_video_path)
    model = fd.vision.segmentation.PaddleSegModel(
        str(model_file), str(params_file), str(config_file), runtime_option=option
    )
    if stack_heatmaps:
        model.postprocessor.apply_softmax = True
    return model, video_input


if __name__ == '__main__':
    show_prediction_frames(
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdmodel',
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdiparams',
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/deploy.yaml',
        '/mnt/DATA/tesi/dataset/dataset_youtube/pallacanestro_trieste/stagione_2019-20_legabasket'
        '/pallacanestro_trieste-virtus_roma/final_cut.mp4',
    )
    # model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    # params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    # config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
