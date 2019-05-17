import keras.backend as K
import glob
from decimal import Decimal
from keras.callbacks import Callback
# from sklearn.metrics import precision_score, recall_score, f1_score
import neptune as npt
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tensorflow.python.ops.metrics import true_positives, false_positives, false_negatives, true_negatives
import tensorflow as tf
import re

def filepattern(pattern, extension, defaulttag='0.0', analysistype=""):
    """
    generates pattern names for efficient exporting of files, great for iterative saving of model parameters as HDF5
    and architechtures as JSON when working with keras

    example call: filepattern('hist_ana_', '.pkl', '5.0', 'convolution_stack) -> hist_ana_5.0convolution_stack.pkl
    above is true provided there is no version tag in the directory higher than 5.0

    :param pattern: defines starting pattern of a file
    :param extension: defines searched file extension
    :param defaulttag: defines default tag if no file that matches pattern is found
    :param analysistype: additional tag for analysis file naming
    :return: returns a filename that follows the same pattern but has higher tag by 0.1
    """
    lst = []
    expression = pattern + '[0-9].[0-9]*' + extension
    # above matches two digit tag separated by dot and accepts any number of additional
    # characters before extension is matched

    for i in glob.glob(expression):
        i = i[(len(pattern)):(len(pattern) + 3)]
        i = Decimal(i)
        lst.append(i.quantize(Decimal('0.1'), rounding='ROUND_DOWN'))
    # finds patterns and saves tags in a list as Decimal objects

    if len(lst) != 0:
        newtag = Decimal('0.1') + Decimal(max(lst))
        newtag = str(newtag)
    else:
        pass  # do nothing (last conditional solves this case already)

    if defaulttag is None:
        defaulttag = Decimal('0.0')
    else:
        defaulttag = Decimal(defaulttag) + Decimal('0.1')

    if len(lst) == 0:
        filename = pattern + str(defaulttag) + analysistype + extension
    else:
        filename = pattern + newtag + analysistype + extension
    return filename


class Precision(Callback):
    """
    Keras Callback. Calculates precision metrics at the end of each epoch.
    """
    def __init__(self):
        super().__init__()
        self.precisions = []

    def on_train_begin(self, logs={}):
        self.precisions = []

    def on_epoch_end(self, epoch, logs={}):
        y_pred = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        y_true = self.validation_data[1]
        precision = precision_score(y_true, y_pred)
        self.precisions.append(precision)
        print("validation set precision at epoch {}: {}".format(epoch, precision))
        return


class Recall(Callback):
    """
    Keras Callback. Calculates recall metrics at the end of each epoch.
    """
    def __init__(self):
        super().__init__()
        self.recalls = []
        # self.X_test = X_test
        # self.y_test = y_test
        self.recalls = []
        self.tps = np.array([])
        self.fns = np.array([])
        self.pos = np.array([])

    def on_batch_begin(self, batch, logs=None):
        tp = np.array([i.eval() for i in true_positives(self.model.targets, self.model.outputs)])
        fn = np.array([i.eval() for i in false_negatives(self.model.targets, self.model.outputs)])
        pos = tp + fn
        self.tps = np.append(self.tps, tp)
        self.fns = np.append(self.fns, fn)
        self.pos = np.append(self.pos, pos)

    def on_epoch_end(self, epoch, logs={}):
        self.recalls = [self.tps, self.fns, self.pos]


class NeptuneMonitor(Callback):
    def __init__(self, batch_size):
        super().__init__()
        self.current_epoch = 0
        self.batch_size = batch_size

    def on_batch_end(self, batch, logs=None):
        x = (self.current_epoch * self.batch_size) + batch
        npt.send_metric('batch end accuracy', x=x, y=logs['acc'])
        npt.send_metric('batch end loss', x=x, y=logs['loss'])

    def on_epoch_end(self, epoch, logs=None):
        npt.send_metric('epoch end accuracy', x=epoch, y=logs['acc'])
        npt.send_metric('epoch end loss', x=epoch, y=logs['loss'])
        npt.send_metric('validation epoch end precision', x=epoch, y=logs['precision'])
        npt.send_metric('validation epoch end recall', x=epoch, y=logs['recall'])
        self.current_epoch += 1


class DebuggingCallback(Callback):
    def __init__(self,):
        self.model.predict

    def on_epoch_end(self, epoch, logs=None):
        print(self.model.outputs)
        print(self.model.targets)

def dict_swap(dictionary):
    """
    swaps keys with values of a dictionary
    load a dict like {'key1': 0, 'key2': 0, 'key3': 1, 'key4': 1}
    get a dict like {0: [key1, key2], 1: [key3, key4]}
    :param dictionary:
    :return:
    """
    out_dict = {}
    for k, v in dictionary.items():
        if v in out_dict.keys():
            if (type(out_dict[v]) is list) and (len(out_dict[v])>0):
                out_dict[v].append(k)
            elif type(out_dict[v]) is not None:
                temp = out_dict[v]
                out_dict[v] = []
                out_dict[v].append(temp)
                out_dict[v].append(k)
        else:
            out_dict[v] = k
    return out_dict


def pool_generator_classes(generator, class_pool_mapping, mode='sparse'):
    """
    FIX DOCSTRINGS!!!!!!!!
    say we have 4 classes that we want to merge into 2; this function will take turn the label stored in generator
    into a label picked from a class_pool_mapping list corresponding to its index; to pool [0, 1, 2, 3] existing labels
    where [0, 1] are supposed to be a new class 0 and [2, 3] a new class 1, you can pass [0, 0, 1, 1]
    as class_pool_mapping; so e.g. class_pool_mapping[2] will turn the label 2 into label 1, class_pool_mapping[1]
    will turn the label 0 into 1 and so on
    :param generator: keras.preprocessing.image.DirectoryIterator object
    :param class_pool_mapping: a dict mapping new labels {old: new}
    :param mode: mode of labelling, should be the same as one used in ImageDataGenerator, default is 'sparse'
    :return:
    """
    # todo - implement mode switch for better scalability
    new_class_list = []

    if mode == 'sparse':
        for idx, label in enumerate(generator.classes):
            if label in class_pool_mapping.keys():
                generator.classes[idx] = class_pool_mapping[label]
            else:
                raise RuntimeError("Missing labels in mapping dict!")
        for k, v in generator.class_indices.items():
            if v in class_pool_mapping.keys():
                generator.class_indices[k] = class_pool_mapping[v]
            else:
                raise RuntimeError("Missing labels in mapping dict!")
        new_class_indices = dict_swap(generator.class_indices)
        for k, v in new_class_indices.items():
            if type(v) is not list:
                pass
            else:
                new_class_indices[k] = ' or '.join(v)
        generator.class_indices = new_class_indices

    else:
        raise RuntimeError("Invalid string passed as 'mode' kwarg!")


def show_architecture(base_model):
    """
    shows all enumerated layers and their names in the model architecture
    :param base_model: base model used in transfer learning to be inspected
    :return: a generator of (idx, layer.name) tuples
    """
    for idx, layer in enumerate(base_model.layers):
        yield (idx, layer.name)


def frosty(layer_iterable, view='len', frost=True):
    """
    freezes/unfreezes layers specified in the iterable passed to the function
    :param layer_iterable: an iterable of Keras layers with .trainable property
    :param view: switches return mode, can be 'len' or 'index'
    :param frost: boolean value, freezes layers if True, unfreezes when False
    :return: returns number of frozen layers if view=='len', if view=='index' returns indices of all frozen layers
    """
    idx_record = []
    for idx, layer in enumerate(layer_iterable):
        assert layer.trainable is not None, "Item passed as layer has no property 'trainable'"
        layer.trainable = ~frost
        idx_record.append(idx)

    if view == 'len':
        return "Number of frozen layers: {}".format(len(idx_record))
    elif view == 'index':
        return "Frozen layers: {}".format(idx_record)
    else:
        pass


def file_train_test_split(path, fmt, split=0.2, random_state=None):
    """
    A function to perform train/test split within a given directory.
    :param path: pathlib.Path object
    :param fmt: format of the files passed, can be list of formats
    :param split: float specifying proportion of test set split
    :param random_state: np.random.seed setting
    :return: two-tuple of pd.DataFrame objects (train, test)
    """
    np.random.seed(random_state)
    cats = list(os.walk(path))[0][1]

    def glob_up(cat):
        if type(fmt) is list:
            globs = []
            for i in fmt:
                globs += glob.glob(str(path / Path(cat) / Path(i)))
            return np.array(globs)
        else:
            return glob.glob(str(path / Path(cat) / Path(fmt)))

    globbed_filenames = {cat: np.array(glob_up(cat)) for cat in cats}  # {'locked': [x.jpg, y.jpg, z.jpg]...}
    cat_lengths = {cat: len(globbed_filenames[cat]) for cat in cats}  # {'locked: 214...}
    # # randomly select equal number of samples from each category
    # # (we're providing a balanced test set even if training dataset is imbalanced)
    test_dict = {}
    train_dict = {}
    for k, v in globbed_filenames.items():
        rand_filenames = np.copy(v)
        np.random.shuffle(rand_filenames)
        split_bound = int(cat_lengths[k]*split)
        test_dict[k] = rand_filenames[:split_bound]
        train_dict[k] = rand_filenames[split_bound:int(cat_lengths[k])]
    return pd.DataFrame.from_dict(train_dict, orient='index', dtype=np.str)\
               .transpose().melt().dropna().rename({'variable': 'y_col', 'value': 'x_col'}, axis=1), \
          pd.DataFrame.from_dict(test_dict, orient='index', dtype=np.str)\
               .transpose().melt().dropna().rename({'variable': 'y_col', 'value': 'x_col'}, axis=1)


def _variable_on_cpu(name, shape, initializer):  # taken from CIFAR10 example on TF webpage
    """Helper to create a Variable stored on CPU memory.

    Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

    Returns:
    Variable Tensor
    """
    with tf.device('/cpu:0'):  # this binds a variable to CPU
        dtype = tf.float32  # and modifies its dtype
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)  # this method gets an existing variable
      # of a given name or creates a new one with that name
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):  # taken from CIFAR10 example on TF webpage
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

    Returns:
    Variable Tensor
    """
    dtype = tf.float32
    var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')  # L2 regularization of weights
        tf.add_to_collection('losses', weight_decay)
    return var


def _activation_summary(x):  # taken from CIFAR10 example on TF webpage
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % 'tower', '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
