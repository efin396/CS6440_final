"""

This file holds the code for training a decision forest on the train data and storing it as a model file for inference

sources:

https://www.tensorflow.org/decision_forests

"""

from tensorflow_decision_forests.keras import (
    pd_dataframe_to_tf_dataset,
    RandomForestModel
)
from models.compile_data import test_df, train_df

MODEL_OUTPUT_PATH = "./df_model/model.keras"

training = pd_dataframe_to_tf_dataset(train_df, label="DEPENDENT")
testing = pd_dataframe_to_tf_dataset(test_df, label="DEPENDENT")

model = RandomForestModel()
model.fit(training)

model.compile(metrics=['accuracy'])
model.evaluate(testing, return_dict=True)

model.save(MODEL_OUTPUT_PATH)