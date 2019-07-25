# Adapted from https://gist.github.com/eavidan/07928337e2859bf8fa607f5693ee4a89#file-tensorflow_serving_rest_client-py

from cnn_toolkit import image_bytestring, load_byte_img
from fubar_CONF import hprm
import numpy as np
import requests


def tf_serving_predict(image,
                       host,
                       port=8501,
                       model_name='fubar',
                       model_version='1',
                       batch_size=1,
                       signature_name='serving_default'):
    """
    function placing calls to the TF Serving Model through REST-API interface
    :param image: path to inference image or image as np.array
    :param host: host address of server with exposed API port
    :param port: API port, default is 8501
    :param model_name: string, model name, default is 'fubar'
    :param model_version: integer, denotes currently served model version
    :param batch_size: batch_size of photos, default is 1
    :param signature_name: model signature_name
    :return: JSON string result
    """
    if isinstance(image, str):
        im = load_byte_img(image_bytestring(image), hprm['INPUT_H'], hprm['INPUT_W']) / 255
    elif isinstance(image, np.ndarray):
        im = image
    else:
        raise TypeError('Invalid image argument!')
    batch = np.repeat(im, batch_size, axis=0).tolist()
    request = {
        "signature_name": signature_name,
        "instances": batch
    }
    response = requests.post(f"http://{host}:{port}/v1/models/{model_name}/versions/{model_version}:predict",
                             json=request)
    # response = requests.post(f"http://localhost:8501/v1/models/fubar/versions/4:predict", json=request)
    result = response.json()
    return result
