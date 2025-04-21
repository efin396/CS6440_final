"""

This code holds the inference endpoints for each model. Each endpoint will take a single line from a dataframe or a numpy array and 
run inference on it based on the desired model.

"""
from typing import List, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow_decision_forests.keras import (
    pd_dataframe_to_tf_dataset,
    RandomForestModel
)
import tf_keras
from fastapi import FastAPI
from pydantic import BaseModel

DF_MODEL_PATH = 'models/decision_forests/df_model'  #TODO: Update with model path
NN_MODEL_PATH = 'models/neural_nets/nn_model/model.keras'  #TODO: Update with model path

# Load the models for inference
dforest = tf_keras.models.load_model(DF_MODEL_PATH)
nnet = tf.keras.models.load_model(NN_MODEL_PATH)

app = FastAPI()

class InputData(BaseModel):
    data: List[List[Any]]

COLUMNS = [
    'MARITAL',
    'RACE',
    'ETHNICITY',
    'GENDER',
    'INCOME',
    'AGE',
    'CHRONIC_PAIN',
    'CHRONIC_MIGRAINE',
    'IMPACTED_MOLARS'
]

@app.post('/inference_df')
def inference_df(payload: InputData):
    try:
        df = pd.DataFrame(payload.data, columns=COLUMNS)
        df = df.astype({
            'MARITAL': 'str',
            'RACE': 'str',
            'ETHNICITY': 'str',
            'GENDER': 'str',
            'INCOME': 'int64',
            'AGE': 'int64',
            'CHRONIC_PAIN': 'int64',
            'CHRONIC_MIGRAINE': 'int64',
            'IMPACTED_MOLARS': 'int64'
        })
        print(df.dtypes)
        print(df.head())
        input = pd_dataframe_to_tf_dataset(df)
        probs = dforest.predict(input)
        return {"predictions": probs.tolist()}
    except Exception as e:
        return {"error": str(e)}

@app.post('/inference_nnet')
def inference_nn(payload: InputData):
    df = pd.DataFrame(payload.data)
    probs = nnet.predict(df)
    return {"predictions": probs.tolist()}
