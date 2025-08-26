"""
ensemble.py
Model ensembling: averaging predictions from multiple models.
"""

import numpy as np

def average_ensemble(predictions_list):
    """
    predictions_list: List of arrays of probabilities from different models
    Returns: averaged probabilities
    """
    return np.mean(np.array(predictions_list), axis=0)