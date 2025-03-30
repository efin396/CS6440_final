"""

This file holds the code for training a decision forest on the train data and storing it as a model file for inference

sources:

https://www.tensorflow.org/decision_forests

"""

import pandas as pd
from tensorflow_decision_forests.keras import (
    pd_dataframe_to_tf_dataset,
    RandomForestModel
)

MODEL_OUTPUT_PATH = "./df_model"
TRAIN_DATA = "../data/train" # TODO: Update this path with actual data
TEST_DATA = "../data/test" #TODO: Update this path with actual data

train_df = pd.read_csv(TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA)

training = pd_dataframe_to_tf_dataset(train_df, label="dependence") #TODO: Update with real label
testing = pd_dataframe_to_tf_dataset(test_df, label="dependence") #TODO: Update with real label

model = RandomForestModel()
model.fit(train_df)

model.compile(metrics=['accuracy'])
model.evaluate(testing, return_dict=True)

model.save(MODEL_OUTPUT_PATH)