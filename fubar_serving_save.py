import tensorflow as tf
import os

model_version = input('Enter model version to be exported as protobuf :')

with open(f'model_partfreeze{model_version}.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights(f'weights_partfreeze{model_version}.h5')

tf.saved_model.save(model, os.getcwd() + f'/{model_version}')
