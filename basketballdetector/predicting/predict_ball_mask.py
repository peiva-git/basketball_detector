"""
This module contains the functions and classes used for mask prediction
"""

import pathlib
import time
import re

from vidgear.gears import CamGear
from vidgear.gears import WriteGear
from statistics import mean

import cv2 as cv
import fastdeploy as fd


class PredictionHandler:
    """
    This class is used to perform predictions starting with an input video and a given model.
    """
    __predictions_target_dir = pathlib.Path.cwd() / 'output'
    __frame_processing_times = []
    __counter = 1
    __YT_URL_REGEX = re.compile(r'https://youtu\.be/.{1,100}')

    def __init__(self,
                 model_file_path: str,
                 params_file_path: str,
                 config_file_path: str,
                 input_video_path: str,
                 use_trt: bool = False):
        """
        The constructor initializes the model and the video stream.
        When one of the available methods is invoked, the video streaming and processing will start.
        This class also takes care of performing the necessary cleanup when a stream is closed.
        :param model_file_path: The path to the model.pdmodel file
        :param params_file_path: The path to the model.pdiparams file
        :param config_file_path: The path to the deploy.yaml file
        :param input_video_path: Input video path, can be a local filepath or a YouTube sharing link
        :param use_trt: Whether to use TensorRT acceleration, if available
        """
        self.__model_file = pathlib.Path(model_file_path)
        self.__params_file = pathlib.Path(params_file_path)
        self.__config_file = pathlib.Path(config_file_path)
        self.__use_trt = use_trt
        if re.match(self.__YT_URL_REGEX, input_video_path) is not None:
            # the input video path is a YouTube link
            print('Start streaming from YouTube video...')
            self.__stream = CamGear(source=input_video_path, stream_mode=True, logging=True).start()
        else:
            self.__stream = CamGear(source=str(pathlib.Path(input_video_path))).start()
        self.__model = self.__setup_model()

    def __setup_model(self) -> fd.vision.segmentation.PaddleSegModel:
        print('Building model...')
        option = fd.RuntimeOption()
        option.use_gpu()
        if self.__use_trt:
            option.use_trt_backend()
            option.set_trt_input_shape(
                'x',
                min_shape=[1, 3, 1024, 2048],
                opt_shape=[4, 3, 1024, 2048],
                max_shape=[8, 3, 1024, 2048]
            )
        model = fd.vision.segmentation.PaddleSegModel(
            str(self.__model_file), str(self.__params_file), str(self.__config_file), runtime_option=option
        )
        return model

    @property
    def predictions_target_directory(self) -> pathlib.Path:
        """
        This method returns the currently set output directory
        :return: The currently set video output directory path
        """
        return self.__predictions_target_dir

    @predictions_target_directory.setter
    def predictions_target_directory(self, predictions_target_dir: pathlib.Path):
        """
        This method sets a new target directory for the produced output video or image sequence
        :param predictions_target_dir: The new output directory
        :return: None
        """
        self.__predictions_target_dir = predictions_target_dir

    def show_prediction_frames(self):
        """
        This method starts the video processing loop.
        After EOF is encountered or if an error occurs,
        the stream is closed and needs to be reopened before using it again.
        This method displays the processed frames without saving them.
        :return: None
        """
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
        """
        This method starts the video processing loop.
        After EOF is encountered or if an error occurs,
        the stream is closed and needs to be reopened before using it again.
        This method saves the processed frames as an image sequence in the specified directory.
        :return: None
        """
        while True:
            frame = self.__stream.read()
            if frame is None:
                break
            output = self.__obtain_prediction(frame)
            cv.imwrite(str(self.__predictions_target_dir / f'frame{self.__counter}.png'), output)
            self.__counter += 1

            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv.destroyAllWindows()
        self.__stream.stop()

    def write_predictions_video(self):
        """
        This method starts the video processing loop.
        After EOF is encountered or if an error occurs,
        the stream is closed and needs to be reopened before using it again.
        This method saves the processed frames as a video in the specified directory.
        :return: None
        """
        writer = WriteGear(output=str(self.__predictions_target_dir / 'predictions.mp4'), compression_mode=False)
        while True:
            frame = self.__stream.read()
            if frame is None:
                break
            output = self.__obtain_prediction(frame)
            writer.write(output)
            self.__counter += 1
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        cv.destroyAllWindows()
        self.__stream.stop()
        writer.close()

    def __obtain_prediction(self, frame):
        start = time.time()
        result = self.__model.predict(frame)
        segmented_image = fd.vision.vis_segmentation(frame, result, weight=0.5)
        end = time.time()
        self.__frame_processing_times.append(end - start)
        print(
            f'Average processing speed: {mean(self.__frame_processing_times):.4f} seconds, {1 / (end - start):.4f} FPS',
            end='\r'
        )
        return segmented_image
