import sys
import os
import glob
import subprocess
import re
import cv2
from PIL import Image
import numpy as np
from fubar_REST import tf_serving_predict
from fubar_CONF import label_dict, labels_of_images_to_be_cropped, tf_s_conf, hprm


def get_lock_image(i):
    #bscript = '#!/bin/bash\n./darknet detect /home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg' + ' ' + \
    #'/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights' + ' ' + '"' + i + '"'
    result = subprocess.run(['./darknet',
                    'detect',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights',
                    i], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print(output)
    m = re.findall('lock:', output)  
    print('we found ' + str(len(m)) + ' lock(s)')
    if len(m) == 1:
       racks = re.findall('rack:', output)
       print('and ' + str(len(racks)) + ' rack(s)')
       lines = len(m) + len(racks)
       for line in output.splitlines():
           if re.findall('lock:', line):
                print('yes')
                print(line)
           else:
               print('no') 

       return {'racks': len(racks)}
    return False


def get_cropped_image(image_path, outfile_draw=None, outfile_crop=None):
    """
    function saving cropped images and images with drawn bounding boxes to specific files
    :param image_path: path to the image on which inference is being run
    :param stdo_blob: intercepted STDOUT from YOLO
    :param outfile_draw: path where bbox drawn outfile should be saved
    :param outfile_crop: path where cropped outfile should be saved
    """
    os.chdir('/home/ubuntu/darknet/AlexeyAB/darknet/')
    result = subprocess.run(['./darknet',
                    'detect',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg',
                    '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights',
                    image_path], stdout=subprocess.PIPE)
    stdo_blob = result.stdout.decode('utf-8')

    im = cv2.imread(image_path)
    H, W, _ = im.shape  # read dimensions of the image
    detected_categories = "|".join(label_dict.keys())
    pattern = f'({detected_categories}): \d+%\n\d+.\d+ \d+.\d+ \d+.\d+ \d+.\d+'
    labels = re.findall(f'({detected_categories}):', stdo_blob)
    confidences = re.findall(r'\d+(?=%)', stdo_blob)
    bbox_dim_list = re.findall(r'\d+.\d+ \d+.\d+ \d+.\d+ \d+.\d+', stdo_blob)
    blobs = zip(labels, confidences, bbox_dim_list)
    data = dict()
    predictions = []
    for idx, i in enumerate(blobs):
        # use regex to find respective parts of the stdout:
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
            prep_img = np.expand_dims(np.array(crop_pil_img.resize((hprm['INPUT_H'], 
                                                           hprm['INPUT_W']))/.255), 
                                                           axis=0)
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
            text = "{}: {:.4f}".format(label, confidence)
            cv2.putText(im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 5, color, 2)
        Image.fromarray(im[:,:,::-1]).save(outfile_draw)  # convert to RGB from BGR and save
    
    return predictions
    

# get_lock_image('/home/ubuntu/darknet/test/IMG_2277.jpg')
   # cropped_images = get_cropped_image()
   # print(cropped_images)
