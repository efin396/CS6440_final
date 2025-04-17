"""

This file contains the training for the dense neural network

sources:
https://www.tensorflow.org/decision_forests

"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import dataframe_image as dfi
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

from models.compile_data import train_df, test_df
import pandas as pd

MODEL_OUTPUT_PATH = "./nn_model/model.keras"

# Before feeding into NN, need to reclassify each categorical variable as a number

val_df, final_test_df = train_test_split(test_df, test_size=0.25, random_state=143)

x_train, y_train = train_df.drop(columns=['DEPENDENT']), train_df['DEPENDENT']
x_test, y_test = final_test_df.drop(columns=['DEPENDENT']), final_test_df['DEPENDENT']
x_val, y_val = val_df.drop(columns=['DEPENDENT']), val_df['DEPENDENT']

y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')

x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)
x_val = pd.get_dummies(x_val)

# Align all sets to the training set
x_val = x_val.reindex(columns=x_train.columns, fill_value=0)
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

input_shape = (x_train.shape[1],)

# Assign imbalanced class weights for better F1 score
class_weights = class_weight.compute_class_weight(
    class_weight={1: 0.75, 0: 0.25},
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = dict(enumerate(class_weights))

model = tf.keras.Sequential([
    tf.keras.Input(input_shape),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Depreciated: Adversarial Regulation isn't supported in modern tf
# adv_config = nsl.configs.make_adv_reg_config(multiplier=0.2, adv_step_size=0.05)
# adv_model = nsl.keras.AdversarialRegularization(model, adv_config=adv_config)

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_val, y_val),
    class_weight=class_weights_dict
)

results = model.evaluate(x_test, y_test)
print(results)

# Confusion matrix
y_true = final_test_df["DEPENDENT"]
y_pred = model.predict(x_test).flatten()
y_pred = (y_pred >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=",", cmap=sns.color_palette("icefire", as_cmap=True))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Dense Neural Network Confusion Matrix")
plt.tight_layout()
plt.savefig('/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/imgs/nn_confusion_matrix.png')
plt.close()

# Classification report
target_names = ['Not Opiod Dependent', 'Opiod Dependent']
cr = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
report_df = pd.DataFrame(cr).transpose()
dfi.export(report_df, '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/imgs/report_nn.png')

model.save(MODEL_OUTPUT_PATH)




