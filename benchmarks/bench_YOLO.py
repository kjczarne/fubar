import os
import subprocess
import re
import cv2
from PIL import Image
import numpy as np
import argparse
import glob
from bench_CONF import label_dict, labels_of_images_to_be_cropped, tf_s_conf, hprm, path_conf
import importlib.util
spec = importlib.util.spec_from_file_location("fubar_rest", "/home/ubuntu/fubar/fubar_REST.py")
fubar_rest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fubar_rest)

spec2 = importlib.util.spec_from_file_location("cnn_toolkit", "/home/ubuntu/fubar/cnn_toolkit.py")
cnn_toolkit = importlib.util.module_from_spec(spec)
spec2.loader.exec_module(cnn_toolkit)


def fubar_benchmark_function(bench_dir, fmt, outfile_draw=None):
    """
    benchmarking function
    :param bench_dir: path to directory with benchmarking images
    :param fmt: str or list specifying globbed format
    :param outfile_draw: name of bbox drawn outfile, will be saved to the same dir
    """
    pred_dict = dict()
    glob_list = []
    if type(fmt) is list:
        for i in fmt:
            glob_list += glob.glob(bench_dir + i)
    else:
        glob_list = glob.glob(bench_dir + fmt)
    for image_path in glob_list:
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
                crop_img = im[y:y + h, x:x + w]  # crop image
                crop_pil_img = Image.fromarray(crop_img[:, :, ::-1])
                # preprocess the image for model input
                prep_img = (np.expand_dims(np.array(crop_pil_img.resize(
                    (hprm['INPUT_H'], hprm['INPUT_W']))), axis=0)) / 255
                # run predictions and append to the prediction list
                predictions.append(fubar_rest.tf_serving_predict(
                    prep_img,
                    host=tf_s_conf['host'],
                    port=tf_s_conf['port'],
                    model_name=tf_s_conf['model_name'],
                    model_version=tf_s_conf['model_version'],
                    batch_size=tf_s_conf['batch_size'],
                    signature_name=tf_s_conf['signature_name']
                ))
        pred_dict[image_path] = {
                                    'classifier_preds': predictions,
                                    'YOLO_preds': {

                                    }
                                 }

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
            Image.fromarray(im[:, :, ::-1]).save(bench_dir+'/'+outfile_draw)  # convert to RGB from BGR and save
        os.chdir(cwd)
    return pred_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image")
    ap.add_argument("-d", "--draw", default='draw.jpg',
                    help="path pointing to where you want to store image with bboxes drawn")
    ap.add_argument("-c", "--crop", default='crop.jpg',
                    help="path pointing to where you want to store cropped image")
    args = vars(ap.parse_args())
    ret = fubar_benchmark_function(args['image'],
                                   outfile_draw=args['draw'],
                                   outfile_crop=args['crop'])
    print(ret)