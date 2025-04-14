"""

This file contains the training for the dense neural network

sources:
https://www.tensorflow.org/decision_forests

"""

import tensorflow as tf
import neural_structured_learning as nsl
import pandas as pd
from models.compile_data import train_df, test_df

MODEL_OUTPUT_PATH = "./nn_model"
TRAIN_DATA = "../data/train" # TODO: Update this path with actual data
TEST_DATA = "../data/test" #TODO: Update this path with actual data


# TODO: Update number of labels

input_shape = (train_df.shape[1],)  # TODO: Update based on number of features
x_train, y_train = train_df.drop(columns=['DEPENDENT']), train_df['DEPENDENT']
x_test, y_test = test_df.drop(columns=['DEPENDENT']), test_df['DEPENDENT']

model = tf.keras.Sequential(
    tf.keras.Input(input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu')
)

# perform adversarial Regulation to protect from noise
adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)

adv_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

adv_model.fit(
    {
        'features': x_train,
        'label': y_train
    },
    batch_size=32,
    epochs=5
)

adv_model.evaluate(
    {
        'features': x_test,
        'label': y_test
    }
)

adv_model.save(MODEL_OUTPUT_PATH)




