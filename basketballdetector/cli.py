import argparse
import pathlib

from basketballdetector import PredictionHandler


def save_predictions_command():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '--stack_heatmaps',
        help='How many multiple heatmaps to stack',
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--use_trt',
        help='Whether to use TensorRT acceleration',
        type=bool,
        required=False,
        default=False
    )

    args = parser.parse_args()
    predictor = PredictionHandler(
        args.model_file,
        args.params_file,
        args.config_file,
        args.input_video,
        args.stack_heatmaps > 0,
        args.use_trt
    )
    predictor.predictions_target_directory = args.target_dir
    if args.stack_heatmaps > 0:
        predictor.number_of_crops = args.stack_heatmaps

    if args.save_mode == 'video':
        predictor.write_predictions_video()
    else:
        predictor.write_image_sequence_prediction()


def display_predictions_command():
    parser = argparse.ArgumentParser()
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
        '--stack_heatmaps',
        help='How many multiple heatmaps to stack',
        type=int,
        required=False,
        default=0
    )
    parser.add_argument(
        '--use_trt',
        help='Whether to use TensorRT acceleration',
        type=bool,
        required=False,
        default=False
    )

    args = parser.parse_args()
    predictor = PredictionHandler(
        args.model_file,
        args.params_file,
        args.config_file,
        args.input_video,
        args.stack_heatmaps > 0,
        args.use_trt
    )
    predictor.predictions_target_directory = args.target_dir
    if args.stack_heatmaps > 0:
        predictor.number_of_crops = args.stack_heatmaps

    predictor.show_prediction_frames()
