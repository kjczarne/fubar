# Adapted from https://gist.github.com/eavidan/07928337e2859bf8fa607f5693ee4a89#file-tensorflow_serving_rest_client-py

from cnn_toolkit import image_bytestring
import numpy as np
import requests

host = 'localhost'
port = '8501'
batch_size = 1
image_path = 'img0.jpg'
model_name = 'fubar'
signature_name = 'serving_default'

image = image_bytestring(image_path)
batch = np.repeat(image, batch_size, axis=0).tolist()

request = {
    "signature_name": signature_name,
    "instances": batch
}
response = requests.post(f"http://{host}:{port}/v1/models/{model_name}:predict", json=request)
print(response)