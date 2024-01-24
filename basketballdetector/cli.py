"""
This module contains the Command Line Interface functions for the `basketballdetector` package.
When the `basketballdetector` package is installed via the
`pip install basketballdetector -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html` command,
the `save-predictions` and `show-predictions` commands are made available in the current environment.

More information on the available arguments can be found in the `basketballdetector` package page.
"""

import argparse
import pathlib

from basketballdetector import PredictionHandler


def save_predictions_command():
    """
    This function is used as an entry point for the train command used by the `basketballdetector` package.
    For a usage example, take a look at the `basketballdetector` package page.
    The accepted command line arguments are `--model_file`, `--params_file`, `--config_file`, `--input_video`,
    `--use_trt`, `--target_dir` and `--save_mode`.
    :return: None
    """
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
    """
    This function is used as an entry point for the train command used by the `basketballdetector` package.
    For a usage example, take a look at the `basketballdetector` package page.
    The accepted command line arguments are `--model_file`, `--params_file`, `--config_file`, `--input_video` and
    `--use_trt`.
    :return: None
    """
    parser = argparse.ArgumentParser()
    __add_common_args(parser)

    args = parser.parse_args()
    predictor = __init_common_predictor(args)

    predictor.show_prediction_frames()


def __add_common_args(parser):
    parser.add_argument(
        '--model_dir',
        help='Directory containing the model.pdmodel, model.pdiparams and config.yaml files',
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
        action=argparse.BooleanOptionalAction,
        required=False,
        default=False
    )
    return parser


def __init_common_predictor(args):
    model_file = pathlib.Path(args.model_dir) / 'model.pdmodel'
    params_file = pathlib.Path(args.model_dir) / 'model.pdiparams'
    config_file = pathlib.Path(args.model_dir) / 'deploy.yaml'
    predictor = PredictionHandler(
        str(model_file),
        str(params_file),
        str(config_file),
        args.input_video,
        args.use_trt
    )
    return predictor
