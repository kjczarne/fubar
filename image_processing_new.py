import sys
import os
import glob
import subprocess
import re
import cv2
from PIL import Image
import numpy as np
import argparse
from fubar_REST import tf_serving_predict
from fubar_CONF import label_dict, labels_of_images_to_be_cropped, tf_s_conf, hprm, path_conf


def fubar_master_function(image_path, outfile_draw=None, outfile_crop=None):
    """
    function saving cropped images and images with drawn bounding boxes to specific files
    :param image_path: path to the image on which inference is being run
    :param stdo_blob: intercepted STDOUT from YOLO
    :param outfile_draw: path where bbox drawn outfile should be saved
    :param outfile_crop: path where cropped outfile should be saved
    """
    cwd = os.getcwd()
    os.chdir(path_conf['yolo_darknet_app'])
    result = subprocess.run(['./darknet',
                    'detect',
                    path_conf['yolo_cfg'],
                    path_conf['yolo_weights'],
                    image_path], stdout=subprocess.PIPE)
    stdo_blob = result.stdout.decode('utf-8')
    print(stdo_blob)
    im = cv2.imread(image_path)
    H, W, _ = im.shape  # read dimensions of the image
    # use regex to find respective parts of the stdout:
    detected_categories = "|".join(label_dict.keys())
    labels = re.findall(f'({detected_categories}):', stdo_blob)
    confidences = re.findall(r'\d+(?=%)', stdo_blob)
    bbox_dim_list = re.findall(r'\d+.\d+ \d+.\d+ \d+.\d+ \d+.\d+', stdo_blob)
    # bbox_dim_list.reverse()
    print(labels)
    print(confidences)
    print(bbox_dim_list)
    blobs = zip(labels, confidences, bbox_dim_list)
    data = dict()
    predictions = []
    for idx, i in enumerate(blobs):
        label = i[0]
        confidence = int(i[1])
        bbox_dims = i[2].split(' ')
        bbox_dims = [float(i) for i in bbox_dims]  # convert bbox_dims to floats
        box = bbox_dims * np.array([W, H, W, H])  # rescale to original image dims
        centerX, centerY, width, height = box.astype("int")
        x = int(centerX - (width / 2))  # top-left corner x coordinate
        y = int(centerY - (height / 2))  # top-left corner y coordinate
        box = (x, y, int(width), int(height))  # format sufficient to draw bounding boxes
        data[idx] = dict(bbox=box, confidence=confidence, label=label)
        x, y, w, h = box
        if label in labels_of_images_to_be_cropped:  # if this is a 'lock"
            crop_img = im[y:y+h, x:x+w]  # crop image
            crop_pil_img = Image.fromarray(crop_img[:,:,::-1])
            if outfile_crop is not None:
                crop_pil_img.save(outfile_crop)  # save image if specified
            # preprocess the image for model input
            prep_img = (np.expand_dims(np.array(crop_pil_img.resize(
                (hprm['INPUT_H'], hprm['INPUT_W']))), axis=0))/255
            # run predictions and append to the prediction list
            predictions.append(tf_serving_predict(
                prep_img,
                host=tf_s_conf['host'],
                port=tf_s_conf['port'],
                model_name=tf_s_conf['model_name'],
                model_version=tf_s_conf['model_version'],
                batch_size=tf_s_conf['batch_size'],
                signature_name=tf_s_conf['signature_name']
            ))

    all_labels = [i['label'] for i in data.values()]
    unique, counts = np.unique(all_labels, return_counts=True)
    label_counts = dict(zip(unique, counts))

    for k, v in label_counts.items():
        print(f'Found {v} instance(s) of {k}')

    if outfile_draw is not None:
        # draw bbox around image:
        for sub_dict in data.values():
            x, y, w, h = sub_dict['bbox']
            confidence = sub_dict['confidence']
            label = sub_dict['label']
            color = label_dict[label]['color']  # set color of the bbox according to label
            cv2.rectangle(im, (x, y), (x + w, y + h), color, 10)
            text = f'{label}: {confidence}%'
            cv2.putText(im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 10)
        Image.fromarray(im[:,:,::-1]).save(outfile_draw)  # convert to RGB from BGR and save
    os.chdir(cwd)
    return predictions
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
	help="path to input image")
    ap.add_argument("-d", "--draw", default='draw.jpg',
	help="path pointing to where you want to store image with bboxes drawn")
    ap.add_argument("-c", "--crop", default='crop.jpg',
	help="path pointing to where you want to store cropped image")
    args = vars(ap.parse_args())
    fubar_master_function(args['image'], 
                          outfile_draw=args['draw'], 
                          outfile_crop=args['crop'])
