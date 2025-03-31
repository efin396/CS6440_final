"""

This code holds the inference endpoints for each model. Each endpoint will take a single line from a dataframe or a numpy array and 
run inference on it based on the desired model.

"""

import numpy as np
import pandas as pd
import tensorflow as tf


DF_MODEL_PATH = '../models/decision_forests/' #TODO: Update with model path
NN_MODEL_PATH = '../models/neural_nets/' #TODO: Update with model path

# Load the models for inference
dforest = tf.keras.models.load_model(DF_MODEL_PATH)
nnet = tf.keras.models.load_model(NN_MODEL_PATH)

def inference_df(data: pd.DataFrame | np.ndarray):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    results: np.ndarray = dforest.predict_proba()
    return results

def inference_nn(data: pd.DataFrame | np.ndarray):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    results: np.ndarray = nnet.predict_proba()
    return results
