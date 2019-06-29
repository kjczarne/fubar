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


def count(directory):
    text_files = glob.glob(directory+'/*.txt')
    dct = dict()
    for i in text_files:
        with open(i, 'r') as f:
            contents = f.read()
        lines = contents.split('\n')
        for n in lines:
            try:
                dct[n.split(' ')[0]].append(i)
            except KeyError:
                dct[n.split(' ')[0]] = [i]
    out_dict = {k: len(v) for k, v in dct.items()}
    return out_dict


def fubar_benchmark_function(thresh_linspace_div=4,
                             iou_thresh_linspace_div=4,
                             metrics=('map'),
                             optimization=('max'),
                             add_metrics=('recall', 'precision', 'TP_', 'FP_', 'FN_'),
                             return_val=None):
    """
    benchmarking function, iterates over bucketed IoU and confidence thresholds and returns optimal
    solution for minimization or maximization of a specific metric
    :param thresh_linspace_div: number of buckets between 0 and 1 to set for confidence threshold
    :param iou_thresh_linspace_div: number of buckets between 0 and 1 to set for IoU threshold
    :param metrics: list of metrics to be used for evaluation, can be:
        * precision
        * recall
        * F1
        * TP
        * FP
        * FN
        * TP_<class name>
        * FP_<class name>
        * FN_<class name>
        * ap_<class name>
        * map
        * precision_<class name>
        * recall_<class name>
    :param optimization: list of functions to be used for evaluation, allows two strings: 'min' and 'max' or any
                         arbitrary function returning numpy array index
                         order must reflect order of metrics being
                         evaluated, e.g. if your metrics=['map', 'FN'] and you want to maximize mAP and minimize
                         false negatives, use ['max', 'min']
    :param return_val: set to anything but None if you want the function to return the data apart from only printing it
    :return: None or final_out dict ({metrics: {confidence_threshold, value}})
    """
    cwd = os.getcwd()
    os.chdir(path_conf['yolo_darknet_app'])
    thresholds = np.linspace(0.05, 0.99, thresh_linspace_div)
    iou_thresholds = np.linspace(0.05, 0.99, iou_thresh_linspace_div)
    print(f"Searching through confidence thresholds: {thresholds}")
    print(f"Searching through IoU thresholds: {iou_thresholds}")
    print('\n')
    patterns = {
        'precision': r'(?<=precision = )\d\.\d+',
        'recall': r'(?<=recall = )\d\.\d+',
        'F1': r'(?<=F1-score = )\d\.\d+',
        'TP': r'(?<=[^\(]TP = )\d+',
        'FP': r'(?<=FP = )\d+(?=,)',
        'FN': r'(?<=[^\(]FN = )\d+',
        'TP_': r'(?<=\(TP = )\d+',
        'FP_': r'(?<=FP = )\d+(?=\))',
        'ap_': r'(?<=ap = )\d+\.\d+',
        'iou_thresh': r'(?<=IoU threshold = )\d+',
        'thresh': r'(?<=thresh = )\d+\.\d+',
        'class_names': r'(?<=name = )\w+',
        'map': r'(?<=\(mAP@\d.\d\d\) = )\d\.\d+'
    }

    category_counts_dct = count(path_conf['yolo_test_set'])
    category_counts = [int(category_counts_dct[i]) for i in sorted(category_counts_dct.keys())]
    # get a sorted list of total detections for each category

    runs_dict = {}
    for u in iou_thresholds:
        for i in thresholds:
            # result = subprocess.run(['./darknet',
            #                          'detector',
            #                          'map',
            #                          path_conf['yolo_obj.data'],
            #                          path_conf['yolo_cfg'],
            #                          path_conf['yolo_weights'],
            #                          '-thresh',
            #                          str(i),
            #                          '-iou-thresh',
            #                          str(u)], stdout=subprocess.PIPE)
            # result = result.stdout.decode('utf-8')
            result = "calculation mAP (mean average precision)...\
    408\
     detections_count = 1276, unique_truth_count = 701\
    class_id = 0, name = lock, ap = 86.82%           (TP = 279, FP = 9)\
    class_id = 1, name = rack, ap = 80.40%           (TP = 225, FP = 23)\
    \
     for thresh = 0.50, precision = 0.94, recall = 0.72, F1-score = 0.81\
     for thresh = 0.50, TP = 504, FP = 32, FN = 197, average IoU = 70.68 %\
    \
     IoU threshold = 50 %, used Area-Under-Curve for each unique Recall\
     mean average precision (mAP@0.50) = 0.836113, or 83.61 %\
    Total Detection Time: 33.000000 Seconds"
            # per-class TP, FP and NP are sorted 0 to n, where n is number of classes

            results = {k: re.findall(v, result) for k, v in patterns.items()}

            # convert types in the results
            # if single element in list change to scalar
            copy_results = {k: v for k, v in results.items()}
            for k, v in results.items():
                repl_list = []
                if type(v) is list:
                    for j in v:
                        try:
                            repl_list.append(int(j))
                        except ValueError:  # converting to int when there's a dot in a string raises a ValueError
                            try:
                                repl_list.append(float(j))
                            except ValueError:
                                repl_list.append(j)  # finally it can be just a string
                v = repl_list
                if len(v) == 1:
                    copy_results[k] = v[0]
                else:
                    copy_results[k] = v

            # FN = all - TP
            # Recall = TP / (TP + FN)
            # Precision = TP / (TP + FP)
            np_all = np.array(category_counts)
            np_fn = np_all - np.array(copy_results['TP_'])
            np_tp = np.array(copy_results['TP_'])
            np_fp = np.array(copy_results['FP_'])
            copy_results['FN_'] = list(np_fn)
            copy_results['recall_'] = list(np_tp/np_all)
            copy_results['precision_'] = list(np_tp/(np_tp + np_fp))
            results = {k: v for k, v in copy_results.items()}  # update results with the copy
            for k, v in results.items():
                if hasattr(v, '__iter__'):
                    for idx, val in enumerate(v):  # [0, 1, 2, 3]
                        if k == 'thresh':
                            copy_results[k] = val  # right now YOLO doesn't allow to separately manipulate thresholds
                                                   # for each class, so we just select the first element of the list
                        else:
                            new_key = k + f'{results["class_names"][idx]}'
                            copy_results[new_key] = val

            runs_dict[(u, i)] = copy_results  # throw in results dict into dict collecting all the runs

    metrics_dict = {k: [] for k in metrics}  # initialize dict with empty lists for metrics
    selected_runs_dict = {k: [] for k in metrics}
    for run, result_dict in runs_dict.items():
        for k, v in result_dict.items():
            if k in metrics:
                metrics_dict[k].append(v)
                selected_runs_dict[k].append(run)
    temp = []
    for i in optimization:  # make min/max strings to correspond to np.argmin/np.argmax
        if i == 'max':
            temp.append(np.argmax)
        elif i == 'min':
            temp.append(np.argmin)
        else:
            if callable(i):
                temp.append(i)
            else:
                raise TypeError('optimization methods should be functions or "min"/"max" strings')
    optimization_dict = dict(zip(metrics, temp))

    final_out = {}
    metrics_to_compare = metrics_dict.keys()
    for metric, list_of_vals in metrics_dict.items():
        func = optimization_dict[metric]
        idx = func(list_of_vals)
        run_id = selected_runs_dict[metric][idx]
        print(f'Confidence threshold {run_id[1]} and IoU threshold {run_id[0]} is optimal with respect to metric {metric}.')
        print(f'Value of metric {metric} for those thresholds is {list_of_vals[idx]}')
        for k in metrics_to_compare:
            if k in add_metrics:
                continue
            elif k == metric:
                continue
            else:
                add_metrics.insert(0, k)
        for k in add_metrics:
            print(f'Value of metric {k} for those thresholds is {runs_dict[run_id][k]}')
        print(f"Mean average precision @ IoU {run_id[0]} is {runs_dict[run_id]['map']}")
        print(f'Function used for evaluation: {func}')
        print('\n\n')
        final_out[metric] = dict(confidence_threshold=thresholds[idx], value=list_of_vals[idx])

    os.chdir(cwd)  # get back to original working directory

    if return_val is not None:
        return final_out
    else:
        return None


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--thresh_div",
                    help="confidence threshold bucket size, default is 4",
                    default=4)
    ap.add_argument("-i", "--iou_div",
                    help="IoU threshold bucket size, default is 4",
                    default=4)
    ap.add_argument('-m', '--metrics', nargs='+',
                    help='metrics keys to be used for optimization',
                    default=['map'])
    ap.add_argument('-o', '--optimization', nargs='+',
                    help='function names to be used for optimization, usually min/max',
                    default=['max'])
    ap.add_argument('-a', '--add_metrics', nargs='+',
                    help='additional metrics to display',
                    default=['recall', 'precision', 'TP_', 'FP_', 'FN_', ])
    args = vars(ap.parse_args())

    ret = fubar_benchmark_function(thresh_linspace_div=int(args['thresh_div']),
                                   iou_thresh_linspace_div=args['iou_div'],
                                   metrics=args['metrics'],
                                   optimization=args['optimization'],
                                   add_metrics=args['add_metrics'],
                                   return_val=None)
    print(ret)
