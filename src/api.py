"""

This code holds the inference endpoints for each model. Each endpoint will take a single line from a dataframe or a numpy array and 
run inference on it based on the desired model.

"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tf_keras
from fastapi import FastAPI


DF_MODEL_PATH = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/models/decision_forests/df_model'  #TODO: Update with model path
NN_MODEL_PATH = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/models/neural_nets/nn_model/model.keras'  #TODO: Update with model path

# Load the models for inference
dforest = tf_keras.models.load_model(DF_MODEL_PATH)
nnet = tf.keras.models.load_model(NN_MODEL_PATH)

app = FastAPI()

@app.post('/inference_df')
def inference_df(data: pd.DataFrame | np.ndarray):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    results: np.ndarray = dforest.predict_proba(data)
    return results

@app.post('/inference_nnet')
def inference_nn(data: pd.DataFrame | np.ndarray):
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    results: np.ndarray = nnet.predict_proba(data)
    return results
