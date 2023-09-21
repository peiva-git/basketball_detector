# Using the PaddleSeg toolbox

The segmentation model has been trained using a customized version of the sample
configuration file for the PPLiteSeg model applied to the 
[Cityscapes dataset](https://www.cityscapes-dataset.com/) found 
[on the PaddleSeg repository](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml).

## Environment setup

Before being able to train the model, you must install
[Paddle](https://github.com/PaddlePaddle/Paddle)
and [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).
It is recommended to do so in a separate environment. Again, you can use
the [provided conda environment](../conda/pp-environment.yml)
by running the following command:
```shell
conda create --name myenv-pp --file pp-environment.yml
```
**Please note** that both the provided environment and the
[Paddle PyPi release](https://pypi.org/project/paddlepaddle-gpu/) currently
require the CUDA Runtime API version 10.2 to run correctly.
If you want a different version, refer to the 
[official documentation](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html).

Also, to avoid unexpected errors, the [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)
package should be built from source using the provided repository,
while being in the myenv-pp environment:
```shell
cd PaddleSeg
pip install -v -e .
```

## Model training

To train the BasketballDetector segmentation model, run:
```shell
cd PaddleSeg
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
--config ../basketballdetector/config/pp_liteseg_stdc1_basketballdetector_1024x512_pretrain-10rancrops.yml \
--do_eval \
--use_vdl \
--save_interval 500
```
The trained models will then be available in the `PaddleSeg/output` directory.
More information on what these options do and on how to visualize the training process
can be found [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/train/train.md).

## Model evaluation

To evaluate the obtained model, run:
```shell
cd PaddleSeg
python tools/val.py \
--config ../basketballdetector/config/pp_liteseg_stdc1_basketballdetector_1024x512_pretrain-10rancrops.yml \
--model_path output/best_model/model.pdparams
```

For additional options refer to the
[official documentation](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/evaluation/evaluate.md).

