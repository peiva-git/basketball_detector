import pathlib
import time
import re

from vidgear.gears import CamGear
from vidgear.gears import WriteGear
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
        np.pad(
            crop[2],
            (
                (crop[1], image.shape[0] - (crop[1] + crop[2].shape[0])),
                (crop[0], image.shape[1] - (crop[0] + crop[2].shape[1])),
                (0, 0)
            ),
            constant_values=127.5
        )
        for crop in random_crops
    ])
    averaged_heatmap = np.mean([np.reshape(result.label_map, result.shape) for result in results], axis=0)
    rescaled_heatmap = averaged_heatmap * 255
    return rescaled_heatmap.astype(np.uint8)


class PredictionHandler:
    __number_of_crops = None
    __image_sequence_target_directory = pathlib.Path.cwd() / 'image_sequence'
    __output_video_target_path = pathlib.Path.cwd() / 'predicted.mp4'
    __frame_processing_times = []
    __counter = 1
    __YT_URL_REGEX = re.compile(r'https://youtu.be/.{1,100}')

    def __init__(self,
                 model_file_path: str,
                 params_file_path: str,
                 config_file_path: str,
                 input_video_path: str,
                 stack_heatmaps: bool = False,
                 use_trt: bool = False):
        self.__model_file = pathlib.Path(model_file_path)
        self.__params_file = pathlib.Path(params_file_path)
        self.__config_file = pathlib.Path(config_file_path)
        self.__stack_heatmaps = stack_heatmaps
        self.__use_trt = use_trt
        if re.match(self.__YT_URL_REGEX, input_video_path) is not None:
            # the input video path is a YouTube link
            print('Start streaming from YouTube video...')
            self.__stream = CamGear(source=input_video_path, stream_mode=True, logging=True).start()
        else:
            self.__stream = CamGear(source=str(pathlib.Path(input_video_path))).start()
        self.__model = self.__setup_model()

    def __setup_model(self):
        print('Building model...')
        option = fd.RuntimeOption()
        option.use_gpu()
        if self.__use_trt:
            option.use_trt_backend()
            option.set_trt_input_shape('x', [1, 3, 1024, 2048])
        model = fd.vision.segmentation.PaddleSegModel(
            str(self.__model_file), str(self.__params_file), str(self.__config_file), runtime_option=option
        )
        if self.__stack_heatmaps:
            model.postprocessor.apply_softmax = True
        return model

    @property
    def number_of_crops(self):
        return self.__number_of_crops

    @number_of_crops.setter
    def number_of_crops(self, number_of_crops):
        self.__number_of_crops = number_of_crops

    @property
    def image_sequence_target_directory(self):
        return self.__image_sequence_target_directory

    @image_sequence_target_directory.setter
    def image_sequence_target_directory(self, target: str):
        self.__image_sequence_target_directory = pathlib.Path(target)

    @property
    def output_video_target_path(self):
        return self.__output_video_target_path

    @output_video_target_path.setter
    def output_video_target_path(self, output_video_target_path):
        self.__output_video_target_path = output_video_target_path

    def show_prediction_frames(self):
        while True:
            frame = self.__stream.read()
            if frame is None:
                break
            output = self.__obtain_prediction(frame)
            cv.imshow('Output', output)
            self.__counter += 1

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv.destroyAllWindows()
        self.__stream.stop()

    def write_image_sequence_prediction(self):
        while True:
            frame = self.__stream.read()
            if frame is None:
                break
            output = self.__obtain_prediction(frame)
            cv.imwrite(str(self.__image_sequence_target_directory / f'frame{self.__counter}.png'), output)
            self.__counter += 1

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv.destroyAllWindows()
        self.__stream.stop()

    def write_predictions_video(self):
        writer = WriteGear(output=str(self.__output_video_target_path), compression_mode=False)
        while True:
            frame = self.__stream.read()
            if frame is None:
                break
            output = self.__obtain_prediction(frame)
            writer.write(output)
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv.destroyAllWindows()
        self.__stream.stop()
        writer.close()

    def __obtain_prediction(self, frame):
        frame_resized = cv.resize(frame, (2048, 1024))
        print(f'Predicting frame {self.__counter}...')
        start = time.time()
        if self.__stack_heatmaps:
            output = get_prediction_with_multiple_heatmaps(
                frame_resized, self.__model, self.__number_of_crops, variance=100
            )
        else:
            output = get_prediction_with_single_heatmap(frame_resized, self.__model)
        end = time.time()
        print(f'Took {end - start} seconds to process frame {self.__counter}')
        self.__frame_processing_times.append(end - start)
        print(f'Average processing speed: {mean(self.__frame_processing_times)} seconds')
        print(f'{1 / (end - start)} FPS')
        return output


if __name__ == '__main__':
    predictor = PredictionHandler(
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdmodel',
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/model.pdiparams',
        '/home/peiva/PycharmProjects/PaddleSeg/output/inference_model/deploy.yaml',
        'https://youtu.be/ou36exQmXjg?si=0iMHyCTBPUzeXFCk',
    )
    predictor.show_prediction_frames()
    # model_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdmodel')
    # params_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'model.pdiparams')
    # config_file = os.path.join('/home', 'ubuntu', 'PaddleSeg', 'output', 'inference_model', 'deploy.yaml')
