import tf_keras
from models.compile_data import test_df, train_df

DF_MODEL_PATH = '/models/decision_forests/df_model'  #TODO: Update with model path
NN_MODEL_PATH = '/models/neural_nets/nn_model/model.keras'  #TODO: Update with model path

# Load the models for inference
dforest = tf_keras.models.load_model(DF_MODEL_PATH)

df = train_df.drop(['DEPENDENT'])
dforest.predict(train_df)