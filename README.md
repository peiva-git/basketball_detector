# Basketball Detector

:basketball: **BasketballDetector** is a deep-learning based tool that enables automatic
ball detection in basketball broadcasting videos.
Currently, the project is still under development.

## Table of contents

1. [Description](#description)
2. [Project requirements](#project-requirements)
3. [Project setup](#project-setup)
   1. [Special requirements](#special-requirements) 
4. [Using the PaddleSeg toolbox](#using-the-paddleseg-toolbox)

## Description

This project uses the 
[PaddleSeg toolkit](https://github.com/PaddlePaddle/PaddleSeg)
to train a [PPLiteSeg model](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/configs/pp_liteseg).
The configuration file used during training can be found 
[here](basketballdetector/config/basketball_detector_pp_liteseg.yml).
Inference is then performed using 
[FastDeploy](https://github.com/PaddlePaddle/FastDeploy).
In the following sections, you will find detailed instructions on how to set up
a working environment and how to use the model to predict the ball location.

## Project requirements

- Python 3.8/3.9/3.10/3.11
- CUDA >= 11.2, cuDNN >= 8.0

A GPU with CUDA capabilities is recommended. Depending on your hardware and/or your OS,
you might need different drivers: check out Nvidia's
[official website](https://www.nvidia.com/Download/index.aspx?lang=en-us).

## Project setup

### Installing the dependencies yourself

All the requirements are listed in the 
[`requirements.txt`](requirements.txt) and [`setup.py`](setup.py) files.
To install all the required dependencies,
you can run one of the following commands from the repository's root directory:
```shell
python -m pip install -r requirements.txt
```
Or: 
```shell
python -m pip install .
```

If you want to install the project in development mode, instead you can run:
```shell
python -m pip install -v -e .
```
More information about what development mode is can be found 
[here](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

### Importing the conda environment

Alternatively, you can import the same
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) 
environment that was used during development.
This environment has been tested under the following conditions:
- Ubuntu 22.04 LTS
- NVIDIA driver version 535.104.05 (installed from the
[CUDA toolkit repository](https://developer.nvidia.com/cuda-12-0-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network))
- Driver API CUDA version 12.2 (installed automatically along with the driver)

Using the provided [conda environment file](conda/pp-fd-environment.yml), run:
```shell
conda create --name myenv-fd --file pp-fd-environment.yml
```

Don't forget to set up the required environment variables as well:
```shell
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
```

You can automate the process of adding the environment variables
to execute automatically each time you activate your
conda environment by running the following commands:
```shell
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

### Special requirements

Since currently only an outdated version of the `fastdeploy-gpu-python` package
is available on [PyPi](https://pypi.org/project/fastdeploy-gpu-python/), you need to
follow [additional steps](https://github.com/PaddlePaddle/FastDeploy#-install-fastdeploy-sdk-with-both-cpu-and-gpu-support)
in order to install the latest version. 
If you're using the provided conda environment, you can simply run the following command:
```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

## Using the PaddleSeg toolbox

The segmentation model has been trained using a customized version of the sample
configuration file for the PPLiteSeg model applied to the 
[Cityscapes dataset](https://www.cityscapes-dataset.com/) found 
[on the PaddleSeg repository](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/configs/pp_liteseg/pp_liteseg_stdc1_cityscapes_1024x512_scale1.0_160k.yml).

### Environment setup

Before being able to train the model, you must install
[Paddle](https://github.com/PaddlePaddle/Paddle)
and [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg).
It is recommended to do so in a separate environment. Again, you can also use
the [provided conda environment](conda/pp-environment.yml)
by running the following command:
```shell
conda create --name myenv-pp --file pp-fd-environment.yml
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

### Model training

To train the BasketballDetector segmentation model, run:
```shell
cd PaddleSeg
export CUDA_VISIBLE_DEVICES=0
python tools/train.py \
--config ../basketballdetector/config/basketball_detector_pp_liteseg.yml \
--do_eval \
--use_vdl \
--save_interval 500
```
The trained models will then be available in the `PaddleSeg/output` directory.
More information on what these options do and on how to visualize the training process
can be found [here](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/docs/train/train.md).


### Model evaluation

To evaluate the obtained model, run:
```shell
cd PaddleSeg
python tools/val.py \
--config ../basketballdetector/config/basketball_detector_pp_liteseg.yml \
--model_path output/best_model/model.pdparams
```