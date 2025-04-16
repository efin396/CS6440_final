"""

This file holds the code for training a decision forest on the train data and storing it as a model file for inference

sources:

https://www.tensorflow.org/decision_forests
https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
https://www.dexplo.org/dataframe_image/#:~:text=Pass%20your%20normal%20or%20styled,save%20it%20as%20an%20image.&text=You%20may%20also%20export%20directly,export%20and%20export_png%20methods%2C%20respectively.

"""

from tensorflow_decision_forests.keras import (
    pd_dataframe_to_tf_dataset,
    RandomForestModel
)
from models.compile_data import test_df, train_df
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)
import dataframe_image as dfi

MODEL_OUTPUT_PATH = "./df_model/"

training = pd_dataframe_to_tf_dataset(train_df, label="DEPENDENT")
testing = pd_dataframe_to_tf_dataset(test_df, label="DEPENDENT")

model = RandomForestModel()
model.fit(training)

model.compile(metrics=['accuracy'])

results = model.evaluate(testing, return_dict=True)
print(results)

# Confusion matrix
y_true = test_df["DEPENDENT"]
y_pred = model.predict(testing)
y_pred = (y_pred >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt=",", cmap=sns.color_palette("icefire", as_cmap=True))
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig('/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/imgs/confusion_matrix.png')
plt.close()

# Classification report
target_names = ['Not Opiod Dependent', 'Opiod Dependent']
cr = classification_report(y_true, y_pred, output_dict=True, target_names=target_names)
report_df = pd.DataFrame(cr).transpose()
dfi.export(report_df, '/Users/willferguson/Downloads/GT Spring 2025/CS 6440/CS6440Project/imgs/report_df.png')

model.save(MODEL_OUTPUT_PATH)