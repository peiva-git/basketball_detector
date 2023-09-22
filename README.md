# Basketball Detector

:basketball: **BasketballDetector** is a deep-learning based tool that enables automatic
ball detection in basketball broadcasting videos.
Currently, the project is still under development.

![Predictions video example](https://media.giphy.com/media/DurYHJy6bj38Hydi7J/giphy.gif)

## Table of contents

1. [Description](#description)
2. [Project requirements](#project-requirements)
3. [Project setup](#project-setup)
   1. [Special requirements](#special-requirements) 
4. [Credits](#credits)

## Description

This project uses the [BasketballTrainer package]() to train a **BasketballDetector** model.
Inference is performed using the [FastDeploy](https://github.com/PaddlePaddle/FastDeploy) API.
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

Alternatively, you can import the same [conda](https://docs.conda.io/projects/conda/en/latest/index.html) 
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
is available on [PyPi](https://pypi.org/project/fastdeploy-gpu-python/), you need to follow
[additional steps](https://github.com/PaddlePaddle/FastDeploy#-install-fastdeploy-sdk-with-both-cpu-and-gpu-support)
in order to install the latest version. 
If you're using the provided conda environment, you can simply run the following command:
```shell
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
Otherwise, check out the installation instructions on the
[official website](https://github.com/PaddlePaddle/FastDeploy#-installation).
Note that **using a conda environment is recommended**.

## Making predictions



## Credits

The model's pre- and post-processing steps are based on the paper
[Real-time CNN-based Segmentation Architecture for Ball Detection in a Single View Setup](https://arxiv.org/abs/2007.11876).
All credits go to its authors.
This project uses [pdoc](https://pdoc.dev/) to generate its documentation. All credits go to its authors.
