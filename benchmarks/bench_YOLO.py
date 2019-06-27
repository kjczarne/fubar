import os
import subprocess
import re
import cv2
from PIL import Image
import numpy as np
import argparse
import sys
import glob
from bench_CONF import label_dict, labels_of_images_to_be_cropped, tf_s_conf, hprm, path_conf
import importlib.util
sys.path.append('/home/ubuntu/fubar')


def fubar_benchmark_function(bench_dir, fmt, outfile_draw=None):
    """
    benchmarking function
    :param bench_dir: path to directory with benchmarking images
    :param fmt: str or list specifying globbed format
    :param outfile_draw: name of bbox drawn outfile, will be saved to the same dir
    """

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