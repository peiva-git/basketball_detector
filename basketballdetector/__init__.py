"""
This is the root package, containing all the necessary tools to predict the basketball location
starting from an input video.

The easiest way to use the trained model to obtain predictions is by using this package's Command Line Interface.
Two commands are made available after the package is installed:
1. save-predictions
2. show-predictions

Both these commands have similar arguments:

| Parameter        | Description                                      | Is Required | Default  | Available for    |
|------------------|--------------------------------------------------|-------------|----------|------------------|
| --model_path     | Path to model.pdmodel model definition file      | Yes         | -        | Both commands    |
| --params_file    | Path to model.pdiparams parameters file          | Yes         | -        | Both commands    |
| --config_file    | Path to deploy.yaml inference configuration file | Yes         | -        | Both commands    |
| --input_video    | Input video path, or a YouTube link              | Yes         | -        | Both commands    |
| --stack_heatmaps | How many additional heatmaps to stack            | No          | 0        | Both commands    |
| --use_trt        | Whether to use TensorRT acceleration             | No          | False    | Both commands    |
| --target_dir     | Prediction data target directory                 | No          | ./output | save-predictions |
| --save_mode      | Choose how to save predictions, video or img-seq | No          | video    | save-predictions |

Example:
```
show-predictions \
--model_file inference_model/model.pdmodel \
--params_file inference_model/model.pdiparams \
--config_file inference_model/deploy.yaml \
--input_video https://youtu.be/yrFjc0Yhos4?si=xadlwEt68Vg2yVec
```
"""

from .predicting import \
    PredictionHandler
