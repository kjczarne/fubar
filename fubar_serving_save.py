import tensorflow as tf
import json
import os

with open('model_partfreeze0.1.json', 'r') as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights('weights_partfreeze0.1.h5')

tf.saved_model.save(model, os.getcwd() + '/1')
