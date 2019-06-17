import glob
from decimal import Decimal
from tensorflow.python.keras.callbacks import Callback
import neptune as npt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import tensorflow as tf
import os
from PIL import Image
from io import BytesIO


def filepattern(pattern, extension, defaulttag='0.0', add_string=""):
    """
    Generates pattern names for efficient exporting of files, great for iterative saving of model parameters as HDF5
    and architechtures as JSON when working with keras

    Example call: filepattern('hist_ana_', '.pkl', '5.0', 'convolution_stack) -> hist_ana_5.0convolution_stack.pkl
    above is true provided there is no version tag in the directory higher than 5.0

    :param pattern: defines starting pattern of a file
    :param extension: defines searched file extension
    :param defaulttag: defines default tag if no file that matches pattern is found
    :param add_string: additional string tag
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
        filename = pattern + str(defaulttag) + add_string + extension
    else:
        filename = pattern + newtag + add_string + extension
    return filename


def get_fresh_weights_and_model(directory, globstring_model, globstring_weights):
    """
    Scans the directory for model and weights files.
    :param directory: string specifying the parent directory to search, use os.getcwd() here to point
    to the current working directory
    :param globstring_model: string specifying the model filename glob pattern
    :param globstring_weights: string specifying the weights filename glob pattern
    :return: 2-tuple where first element is the path to model architecture, second to weights
    """
    model_names = glob.glob(directory + '/' + globstring_model) 
    model_names.sort(reverse=True)
    weight_names = glob.glob(directory + '/' + globstring_weights)
    weight_names.sort(reverse=True)
    return model_names[0], weight_names[0]


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
        npt.send_metric('validation accuracy', x=epoch, y=logs['val_acc'])
        npt.send_metric('epoch end loss', x=epoch, y=logs['loss'])
        npt.send_metric('epoch end validation loss', x=epoch, y=logs['val_loss'])
        npt.send_metric('training precision', x=epoch, y=logs['precision'])
        npt.send_metric('training recall', x=epoch, y=logs['recall'])
        npt.send_metric('validation precision', x=epoch, y=logs['val_precision'])
        npt.send_metric('validation recall', x=epoch, y=logs['val_recall'])
        self.current_epoch += 1


def dict_swap(dictionary):
    """
    Swaps keys with values of a dictionary
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
    say we have 4 classes that we want to merge into 2; this function will take turn the label stored in generator
    into a label picked from a class_pool_mapping dict; to pool [0, 1, 2, 3] existing labels
    where [0, 1] are supposed to be a new class 0 and [2, 3] a new class 1, you can pass {0:0, 1:0, 2:1, 3:1}
    as class_pool_mapping;
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
        raise RuntimeError("Invalid string passed as 'mode' kwarg!, Non-sparse modes not implemented yet!")


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


def glob_up(path, cat, fmt):
    """
    Function to glob different category folders
    :param path: path to main directory
    :param cat: category string
    :param fmt: format of the files passed, can be list of formats
    :return: list of paths
    """
    if type(fmt) is list:
        globs = []
        for i in fmt:
            globs += glob.glob(str(path / Path(cat) / Path(i)))
        return np.array(globs)
    else:
        return glob.glob(str(path / Path(cat) / Path(fmt)))


def file_train_test_split(path, fmt, split=0.2, random_state=None, ignored_directories=None):
    """
    A function to perform train/test split within a given directory.
    :param path: pathlib.Path object
    :param fmt: format of the files passed, can be list of formats
    :param split: float specifying proportion of test set split
    :param random_state: np.random.seed setting
    :param ignored_directories: list of directories to ignore while reading the main dir
    :return: two-tuple of pd.DataFrame objects (train, test)
    """
    np.random.seed(random_state)
    cats = list(os.walk(path))[0][1]
    try:
        for idx, el in enumerate(cats):
            if el in ignored_directories:
                cats.remove(idx)
    except ValueError:
        pass
    globbed_filenames = {cat: np.array(glob_up(path, cat, fmt)) for cat in cats}  # {'locked': [x.jpg, y.jpg, z.jpg]...}
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

# -------------------------------
# Code below is used for WIT tool |
# -------------------------------

def tf_example_generator(paths_dataframe, x_col='x_col', y_col='y_col'):
    """
    A helper function generating tf.Example instances from input data
    :param paths_dataframe: pd.DataFrame as returned by custom file_train_test_split function
    :param x_col: string specifying the column with pathname
    :param y_col: string specifying the column with labels
    :return: list of tf.Example objects
    """
    examples = []
    unique_labels = set(paths_dataframe[y_col])
    dict_mapping = dict()
    for i in zip(list(unique_labels), range(len(unique_labels))):
        dict_mapping[i[0]] = i[1]
    labels = []
    for i in paths_dataframe[y_col]:
        labels.append(dict_mapping[i])
    for i in paths_dataframe[x_col]:
        example = tf.train.Example()
        with open(i, 'rb') as f:
            im = Image.open(f)
            buffer = BytesIO()
            im.save(buffer, format='JPEG')
            image_bytes = buffer.getvalue()
            example.features.feature['image/encoded'].bytes_list.value.append(image_bytes)
        examples.append(example)
    return examples, labels


def load_byte_img(im_bytes, IMAGE_H, IMAGE_W):
        buf = BytesIO(im_bytes)
        im = np.array(Image.open(buf).resize((IMAGE_H, IMAGE_W)), dtype=np.float64) / 255.
        return np.expand_dims(im, axis=0)


def custom_predict(examples_to_infer, model):
    ims = [load_byte_img(ex.features.feature['image/encoded'].bytes_list.value[0]) 
         for ex in examples_to_infer]
    preds = model.predict(np.array(ims))
    return preds
