"""

This file contains the training for the dense neural network

sources:
https://www.tensorflow.org/decision_forests

"""

import tensorflow as tf
import neural_structured_learning as nsl
import pandas as pd

MODEL_OUTPUT_PATH = "./nn_model"
TRAIN_DATA = "../data/train" # TODO: Update this path with actual data
TEST_DATA = "../data/test" #TODO: Update this path with actual data

train = pd.read_csv(TRAIN_DATA).to_numpy()
test = pd.read_csv(TEST_DATA).to_numpy()

# TODO: Update number of labels

input_shape = (1,) # TODO: Update based on number of features
x_train, y_train = None, None
x_test, y_test = None, None

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




