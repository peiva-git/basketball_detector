"""
This module contains the Command Line Interface functions for the project.
"""

import argparse
import pathlib

from basketballdetector import PredictionHandler


def save_predictions_command():
    parser = argparse.ArgumentParser()
    __add_common_args(parser)
    parser.add_argument(
        '--target_dir',
        help='Prediction data target directory',
        type=str,
        required=False,
        default=pathlib.Path.cwd() / 'output'
    )
    parser.add_argument(
        '--save_mode',
        help='Choose how the predictions will be saved',
        choices=['video', 'img-seq'],
        type=str,
        required=False,
        default='video'
    )

    args = parser.parse_args()
    predictor = __init_common_predictor(args)

    predictor.predictions_target_directory = pathlib.Path(args.target_dir)
    if args.save_mode == 'video':
        predictor.write_predictions_video()
    else:
        predictor.write_image_sequence_prediction()


def display_predictions_command():
    parser = argparse.ArgumentParser()
    __add_common_args(parser)

    args = parser.parse_args()
    predictor = __init_common_predictor(args)

    predictor.show_prediction_frames()


def __add_common_args(parser):
    parser.add_argument(
        '--model_file',
        help='model.pdmodel model definition file path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--params_file',
        help='model.pdiparams model parameters file path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--config_file',
        help='deploy.yaml inference configuration file path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--input_video',
        help='Input video path, could be a video filename or a YouTube link',
        type=str,
        required=True
    )
    parser.add_argument(
        '--use_trt',
        help='Whether to use TensorRT acceleration',
        type=bool,
        required=False,
        default=False
    )
    return parser


def __init_common_predictor(args):
    predictor = PredictionHandler(
        args.model_file,
        args.params_file,
        args.config_file,
        args.input_video,
        args.use_trt
    )
    return predictor
