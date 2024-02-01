Three commands are made available after the package is installed via `pip install basketballdetector`:
1. save-predictions
2. show-predictions
3. test-speed

The `save-predictions` and `show-predictions` commands have similar arguments:

| Parameter        | Description                                                                                     | Is Required | Default  | Available for    |
|------------------|-------------------------------------------------------------------------------------------------|-------------|----------|------------------|
| --model_dir      | Path to the directory containing the `model.pdmodel`, `model.pdiparams` and `config.yaml` files | Yes         | -        | Both commands    |
| --input_video    | Input video path, or a YouTube link                                                             | Yes         | -        | Both commands    |
| --use_trt        | Whether to use TensorRT acceleration                                                            | No          | False    | Both commands    |
| --target_dir     | Prediction data target directory                                                                | No          | ./output | save-predictions |
| --save_mode      | Choose how to save predictions, video or img-seq                                                | No          | video    | save-predictions |

An example:
```shell
show-predictions \
--model_dir inference_model/ \
--input_video https://youtu.be/yrFjc0Yhos4?si=xadlwEt68Vg2yVec
```

The `test-speed` command has the same arguments as the `show-predictions` command.
It is used exclusively for testing, since it does nothing with the produced output.
