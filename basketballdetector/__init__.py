"""
This is the root package of the **BasketballDetector project**,
containing all the necessary tools to predict the basketball location
starting from an input video.

The easiest way to use the trained model to obtain predictions is by using this package's Command Line Interface.
More information on how to obtain the `model.pdmodel`, `model.pdiparams` and `deploy.yaml` files can be found in
the [BasketballTrainer repository](https://github.com/peiva-git/basketball_trainer).

.. include:: ./cli.md

"""

from .predicting import \
    PredictionHandler
