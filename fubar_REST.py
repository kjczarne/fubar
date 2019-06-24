# Adapted from https://gist.github.com/eavidan/07928337e2859bf8fa607f5693ee4a89#file-tensorflow_serving_rest_client-py

from cnn_toolkit import image_bytestring, load_byte_img
from fubar_preprocessing import hprm
import numpy as np
import requests

host = 'localhost'
port = '8501'
batch_size = 1
image_path = 'img0.jpg'
model_name = 'fubar'
model_version = '4'
signature_name = 'serving_default'

image = load_byte_img(image_bytestring(image_path), hprm['INPUT_H'], hprm['INPUT_W'])
batch = np.repeat(image, batch_size, axis=0).tolist()

request = {
    "signature_name": signature_name,
    "instances": batch
}
# response = requests.post(f"http://{host}:{port}/v1/models/{model_name}/{model_version}:predict", json=request)
response = requests.post(f"http://localhost:8501/v1/models/fubar/4:predict", json=request)
print(response)