import os
import subprocess
import re
import cv2
from PIL import Image
import numpy as np
import argparse
import sys
import glob
sys.path.append('/home/ubuntu/fubar')
# sys.path.append('/Users/kjczarne/Desktop/Coding/Python/DSR/fubar')
from fubar_CONF import label_dict, labels_of_images_to_be_cropped, tf_s_conf, hprm, path_conf



def fubar_benchmark_function(thresh_linspace_div=10, iou_thresh=0.5):
    """
    benchmarking function
    :param thresh_linspace_div: number of buckets between 0 and 1 to set for confidence threshold
    :param iou_thresh: iou threshold for
    """
    cwd = os.getcwd()
    os.chdir(path_conf['yolo_darknet_app'])
    thresholds = np.linspace(0, 1, thresh_linspace_div)
    for i in thresholds:
        # result = subprocess.run(['./darknet',
        #                          'detector',
        #                          'map',
        #                          path_conf['yolo_obj.data'],
        #                          path_conf['yolo_cfg'],
        #                          path_conf['yolo_weights'],
        #                          '-thresh',
        #                          i], stdout=subprocess.PIPE)
        stdo_blob = result.stdout.decode('utf-8')

        # per-class TP, FP and NP are sorted 0 to n, where n is number of classes
        patterns = {
            'precision': r'precision = \d\.\d+',
            'recall': r'recall = \d\.\d+',
            'f1': r'F1-score = \d\.\d+',
            'TP': r'[^\(]TP = \d+',
            'FP': r'[^\(]FP = \d+',
            'FN': r'[^\(]FN = \d+',
            'TP_per_class': 'r\(TP = \d+',
            'FP_per_class': r'\(FP = \d+',
            'ap_per_class': r'ap = .\d+\.\d+',
            'iou_thresh': r'IoU threshold = \d+',
            'thresh': r'thresh = \d+\.\d+',
            'class_names': r'name = .\w+',
            'map': r'\(mAP@\d\.\d+\) = \d\.\d+'
        }
        # re.findall()
        # results = {
        #     'precision': r'precision = \d\.\d+',
        #     'recall': r'recall = \d\.\d+',
        #     'f1': r'precision = \d\.\d+',
        #     'TP':,
        #     'FP':,
        #     'FN':,
        #     'TP_per_class':,
        #     'FP_per_class':,
        #     'FN_per_class':,
        # }
    return patterns


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--thresh", required=True,
                    help="confidence threshold bucket size, default is 10")
    ap.add_argument("-i", "--iou_thresh", default='draw.jpg',
                    help="IoU threshold to check mAP at, default is 0.5")
    args = vars(ap.parse_args())
    ret = fubar_benchmark_function(args['thresh'], args['iou_thresh'])
    print(ret)