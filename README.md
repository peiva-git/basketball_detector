# Basketball detector

:basketball: **Basketball detector** is a deep-learning based tool that enables automatic
ball detection in basketball videos.
Currently, the project is still under development.

## Project requirements

- Python 3.8/3.9/3.10/3.11
- CUDA >= 11.2, cuDNN >= 8.0

## Project setup

### Install the dependencies yourself

All the requirements are listed in the `requirements.txt` and `setup.py` files.
To install all the required dependencies,
you can run one of the following commands from the repository root directory:
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

### Import conda environment

Exported [conda](https://docs.conda.io/projects/conda/en/latest/index.html) 
environments will be made available for easier installation.

