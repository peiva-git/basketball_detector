# Basketball Detector

[![build and deploy docs](https://github.com/peiva-git/basketball_detector/actions/workflows/docs.yml/badge.svg)](https://github.com/peiva-git/basketball_detector/actions/workflows/docs.yml)
[![build and test GPU](https://github.com/peiva-git/basketball_detector/actions/workflows/build-and-test-gpu.yml/badge.svg)](https://github.com/peiva-git/basketball_detector/actions/workflows/build-and-test-gpu.yml)
[![build and test CPU](https://github.com/peiva-git/basketball_detector/actions/workflows/build-and-test-cpu.yml/badge.svg)](https://github.com/peiva-git/basketball_detector/actions/workflows/build-and-test-cpu.yml)
![License](https://img.shields.io/github/license/peiva-git/basketball_detector)

:basketball: **BasketballDetector** is a deep-learning based tool that enables automatic
ball detection in basketball broadcasting videos.
Currently, the project is still under development.

<img src="https://media.giphy.com/media/DurYHJy6bj38Hydi7J/giphy.gif" alt="Detections video example" width="100%" height="400px"/>

The complete video is available [here](https://youtu.be/jhQOChtrPWg)

## Table of contents

1. [Description](#description)
2. [Project requirements](#project-requirements)
3. [Project setup](#project-setup)
4. [Credits](#credits)

## Description

This project uses the [BasketballTrainer package](https://github.com/peiva-git/basketball_trainer)
to train a **BasketballDetector** model.
This repository provides all the necessary tools to perform inference on the trained model using the 
[FastDeploy](https://github.com/PaddlePaddle/FastDeploy) API.
In the following sections, you will find detailed instructions on how to set up
a working environment and how to use one a trained model to predict the ball location.

## Project requirements

- Python 3.8/3.9/3.10/3.11
- CUDA >= 11.2, cuDNN >= 8.0

A GPU with CUDA capabilities is recommended. Depending on your hardware and/or your OS,
you might need different drivers: check out Nvidia's
[official website](https://www.nvidia.com/Download/index.aspx?lang=en-us).

## Project setup

**The recommended approach** is to use one of the two provided
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) environments made available [here](conda),
one CPU based and the second GPU based.
By default, the GPU based environment uses the `cudatoolkit=11.2` and `cudnn=8.0` conda packages.
**If you want to use a different CUDA version**,
refer to the [official documentation](https://github.com/PaddlePaddle/FastDeploy/tree/develop).

All the requirements are listed in the 
[`requirements.txt`](requirements.txt) file.
To create a new conda environment meeting all the required dependencies, run the following command:
```shell
conda create --name myenv-fd --file fd-gpu.yml
```

In case you're using the GPU version, don't forget to set up the required environment variables as well:
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

These environments have been tested under the following conditions:
- Ubuntu 22.04 LTS
- NVIDIA driver version 535.104.05 (installed from the
[CUDA toolkit repository](https://developer.nvidia.com/cuda-12-0-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network))
- Driver API CUDA version 12.2 (installed automatically along with the driver)

## Making predictions

For information on how to use package to make predictions with the model, refer to the
[official documentation](https://peiva-git.github.io/basketball_detector/).

## Credits

The model's pre- and post-processing steps are based on the paper
[Real-time CNN-based Segmentation Architecture for Ball Detection in a Single View Setup](https://arxiv.org/abs/2007.11876).
All credits go to its authors.
This project uses [pdoc](https://pdoc.dev/) to generate its documentation. All credits go to its authors.
