"""

This code holds the inference endpoints for each model. Each endpoint will take a single line from a dataframe or a numpy array and 
run inference on it based on the desired model.

"""
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import tf_keras
from fastapi import FastAPI
from pydantic import BaseModel

DF_MODEL_PATH = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/models/decision_forests/df_model'  #TODO: Update with model path
NN_MODEL_PATH = '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/models/neural_nets/nn_model/model.keras'  #TODO: Update with model path

# Load the models for inference
dforest = tf_keras.models.load_model(DF_MODEL_PATH)
nnet = tf.keras.models.load_model(NN_MODEL_PATH)

app = FastAPI()

class InputData(BaseModel):
    data: List[List[float]]

@app.post('/inference_df')
def inference_df(payload: InputData):
    df = pd.DataFrame(payload.data)
    probs = dforest.predict(df)
    return {"predictions": probs.tolist()}

@app.post('/inference_nnet')
def inference_nn(payload: InputData):
    df = pd.DataFrame(payload.data)
    probs = nnet.predict(df)
    return {"predictions": probs.tolist()}
