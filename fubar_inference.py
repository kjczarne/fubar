import tensorflow as tf
import numpy as np
from PIL import Image
import glob
from cnn_toolkit import glob_up
from fubar_preprocessing import file_formats, hprm
import json

# ----------------
# RE-CREATE MODEL |
# ----------------
with open(input('Enter filename with model architecture: '), 'r') as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights(input('Enter filename with model weights: '))
# ---------------------------------------------------------------------------------------------------------------------

# -------------------------------------------
# LISTEN TO A DIRECTORY WITH INFERENCE IMAGES|
# -------------------------------------------
# glob images in results folder
list_of_inference_images = glob_up('/home/ubuntu/darknet/AlexeyAB/darknet/result_img/', 'inference', file_formats)
# user_json = json.loads('/home/ubuntu/darknet/AlexeyAB/darknet/result_img/users.json')
            # A JSON mapping users to their respective photos (as path) in the inference directory
            # keeping count of their properly locked bikes as well

prediction_dict = dict()
for i in list_of_inference_images:
    im = Image.open(i)
    # rescale and resize the image according to model hyperparameters
    im = tf.cast(im.resize((hprm['INPUT_H'], hprm['INPUT_W']))/.255, tf.float32)
    print(model.predict(im))
    # unique_users = []
    # for k, v in user_json.items():
    #     if v == i:
    #         username=v
    #     else:
    #         raise RuntimeError('User does not exist!')
    # prediction_dict[username] = model.predict(im)
# ---------------------------------------------------------------------------------------------------------------------
