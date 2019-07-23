import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
sys.path.append('/home/ubuntu/fubar')
from cnn_toolkit import contiguous_true_vs_predicted, get_fresh_weights_and_model, filepattern


def roc_curve_data(model,
                   positive_class,
                   negative_class,
                   tts):
    """
    saves roc curve data to 3 separate text files (fpr, tpr and thresholds)
    :param model: Model instance implementing predict method
    :param positive_class: string specifying positive class name
    :param negative_class: string specifying negative class name
    :param tts: test pd.DataFrame from file_train_test_split
    :return:
    """

    y_true, y_pred = contiguous_true_vs_predicted(
        model,
        positive_class,
        negative_class,
        tts=tts
    )

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    print(f'AUC is {auc(fpr, tpr)}')

    np.savetxt(filepattern('fpr_', '.txt'), fpr)
    np.savetxt(filepattern('tpr_', '.txt'), tpr)
    np.savetxt(filepattern('thresholds_', '.txt'), thresholds)


def load_roc_curve(fpr_txt=None, tpr_txt=None, thresholds_txt=None):

    def sub(arg, strg):
        if arg is None:
            res = np.loadtxt(filepattern(strg, '.txt'))
        else:
            res = np.loadtxt(strg)
        return res

    fpr = sub(fpr_txt, 'fpr_')
    tpr = sub(tpr_txt, 'tpr_')
    thresholds = sub(thresholds_txt, 'thresholds_')

    return fpr, tpr, thresholds


if __name__ == '__main__':

    cwd = os.getcwd()
    os.chdir('/home/ubuntu/fubar')

    m, w = get_fresh_weights_and_model(os.getcwd(), 'model_allfreeze*', 'weights_allfreeze*')
    with open(m, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(w)

    _, t = get_fresh_weights_and_model(os.getcwd(), 'train*.csv', 'test*.csv')
    # above function can be also used to get file names of the freshest train and test sets
    test_set = pd.read_csv(t)

    roc_curve_data(model, 'locked', 'freelocked', test_set)

    os.chdir(cwd)