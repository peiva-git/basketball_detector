"""
This package contains all the modules used to perform mask prediction.
While the `basketballdetector.predicting.predict_ball_mask.get_prediction_with_single_heatmap`
and `basketballdetector.predicting.predict_ball_mask.get_prediction_with_multiple_heatmaps` functions
simply produce an output from a single image, the `basketballdetector.predicting.predict_ball_mask.PredictionHandler`
class encompasses the whole prediction process starting from an input video.
"""

from .predict_ball_mask import \
    get_prediction_with_single_heatmap, get_prediction_with_multiple_heatmaps, \
    PredictionHandler
