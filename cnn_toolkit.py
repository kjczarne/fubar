import glob
from tensorflow.python.keras.callbacks import Callback
import neptune as npt
import numpy as np
import pandas as pd
import os
import re
from pathlib import Path
import tensorflow as tf
from PIL import Image
from io import BytesIO


def filepattern(pattern, extension, defaulttag='0', add_string=""):
    """
    Generates pattern names for efficient exporting of files, great for iterative saving of model parameters as HDF5
    and architechtures as JSON when working with keras

    Example call: filepattern('hist_ana_', '.pkl', '5', 'convolution_stack) -> hist_ana_5convolution_stack.pkl
    above is true provided there is no version tag in the directory higher than 5

    :param pattern: defines starting pattern of a file
    :param extension: defines searched file extension
    :param defaulttag: defines default tag if no file that matches pattern is found
    :param add_string: additional string tag
    :return: returns a filename that follows the same pattern but has higher tag by 1
    """
    expression = pattern + '[0-9]+' + extension
    # above matches integer tag and accepts any number of additional
    # characters before extension is matched

    globbed = glob.glob(expression)

    lst = re.findall(r'[0-9]+(?=\.)', ''.join(globbed))

    if defaulttag is None:
        defaulttag = '0'
    else:
        defaulttag = str(int(defaulttag) + 1)

    if len(lst) != 0:
        newtag = 1 + max(lst)
        newtag = str(newtag)
        filename = pattern + newtag + add_string + extension
    else:
        filename = pattern + str(defaulttag) + add_string + extension

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


# Taken from https://epcsirmaz.blogspot.com/2017/06/display-sample-predictions-during.html
# Function to display the target and prediciton
def make_pred_output_callback(model, data_iterator, batch_size):
    def testmodel(epoch, logs):
        predx, predy = next(data_iterator)

        predout = model.predict(
            predx,
            batch_size=batch_size
        )

        print("Target\n")
        print(predy)
        print("Prediction\n")
        print(predout)
    return testmodel

# -----------------------------------------------------------------------------------------

# -------------------------------
# Code below is used for WIT tool |
# -------------------------------

# Taken from TensorFlow TFRecord tutorial ----------------------------------------------
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# ----------------------------------------------------------------------------------------


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


def image_bytestring(path_to_image):
    with open(path_to_image, 'rb') as f:
        return f.read()


def image_example(image_string, label, extra_features=None):
    """
    taken from TensorFlow TFRecords tutorial
    :param image_string: image bytestring
    :param label: image label (integer)
    :return: tf.Example
    """
    image_shape = tf.image.decode_jpeg(image_string).shape
    feature = {
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    if extra_features is None:
        pass
    else:
        for k, v in extra_features.items():
            feature[k] = v
    return tf.train.Example(features=tf.train.Features(feature=feature))


def labels_to_integers(label_list):
    """
    converts a list of string labels to integer list
    :param label_list: list of labels (str)
    :return: list
    """
    mapping = {v:k for k, v in enumerate(list(set(label_list)))}
    return [mapping[i] for i in label_list]


def write_tfrecord(df, 
                   image_path_series='x_col', 
                   label_series_name='y_col', 
                   cols_to_ignore=[],  
                   outfile='images.tfrecord',
                   verbose=True):
    """
    writes TFRecord files
    :param df: pd.DataFrame with at least 2 columns containing paths to images and labels
    :param image_path_series: string naming the column with paths, default is 'x_col'
    :param label_series_name: string naming the column with labels, default is 'y_col'
    :param cols_to_ignore: list of columns from the DataFrame to not treat as features and ignore them
    :param outfile: path to output TFRecord file
    :param verbose: verbose mode, True by default
    :return: None
    """
    # truncate df, put label separately, remove non-feature columns
    cols = list(df.columns)
    
    for i in cols_to_ignore:
        cols.remove(i)
    
    sub_df = df.loc[:, cols]
    examples = []
    sub_df[label_series_name] = labels_to_integers(df[label_series_name])
    for i in sub_df.iterrows():
        series = i[1]  # underlying pd.Series
        im_bytes = image_bytestring(series[image_path_series])
        label = series[label_series_name]
        extra_feature_names = list(series.keys())
        extra_feature_names.remove(label_series_name)
        extra_feature_names.remove(image_path_series)
        series = series.loc[extra_feature_names]
        # ignore image path and label for extra features
        extra_features = dict()
        for k, ft in series.items():
            if type(ft) == int:
                val = _int64_feature(ft)
            elif type(ft) == object:
                val = _bytes_feature(ft)
            elif type(ft) == float:
                val = _float_feature(ft)
            else:
                raise ValueError('Unsupported data type!')
            extra_features[k] = val
        im_example = image_example(im_bytes, label, extra_features)
        examples.append(im_example)
        if verbose:
            print(f'Generated {len(examples)} examples')
    
    def write(outfile):
        with tf.io.TFRecordWriter(outfile) as writer:
            for example in examples:
                writer.write(example.SerializeToString())
    
    write(outfile)


def _parse_image_function(example_proto, additional_features=None):
    """
    taken from TensorFlow TFRecords tutorial
    :param example_proto: tf.Example
    :return: parsed binary data, human readable
    """
    image_feature_desc = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    if additional_features is None:
        pass
    else:
        for k, v in additional_features.items():
            image_feature_desc[k] = v
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_desc)


def parse_image_dataset(raw_image_dataset):
    """
    transforms raw binary data in a tf.Dataset to human readable form
    :param raw_image_dataset: tf.Dataset with raw binary data
    :return: human readable tf.Dataset instance
    """
    return raw_image_dataset.map(_parse_image_function)


def serialize_model(model_file, model_version='1', weights_file=None):
    """
    saves model as a protobuf compatible with TensorFlow Serving
    :param model_file: path to JSON or HDF5 file with the model
    :param model_version: string specifying model version name
    :param weights_file: path to optional HDF5 file with weights
    :return: None
    """
    if weights_file is None:
        model = tf.keras.models.load_model(model_file)
    else:
        with open(model_file, 'r') as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights(weights_file)
    tf.saved_model.save(model, os.getcwd() + '/' + model_version)
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------
# HELPER FUNCTIONS FOR MODEL EVALUATION |   
# --------------------------------------


def true_vs_predicted(model, positive_class, negative_class, base_path, accepted_file_formats):
    """
    Function for getting bulk predictions of binary classes.
    :param model: Model or Sequential instance implementing predict method
    :param positive_class: string, folder name of class defined as 1
    :param negative_class: string, folder name of class defined as 0
    :param base_path: string, base path to the folder containing category subfolders
    :param accepted_file_formats: list, glob-like file formats e.g. ['*.jpg']
    :return: 4-tuple of np.arrays containing pred values of class 1, true values of class 1,
             same respectively for class 0
    """
    preds=np.empty((1,0))
    glob_list = []
    
    positive_preds = []
    negative_preds = []
    
    for i in accepted_file_formats:
        glob_list += glob.glob(base_path+'/'+positive_class+'/'+i)
    for i in glob_list:
        im = Image.open(i)
        preds = np.append(preds, model.predict((np.expand_dims(np.array(im.resize((299,299))), axis=0))/255), axis=1)
    positive_preds = preds.squeeze()
    positive_ground_truth = np.array([1 for i in preds.squeeze()])
    
    preds=np.empty((1,0))
    glob_list = []
    for i in accepted_file_formats:
        glob_list += glob.glob(base_path+'/'+negative_class+'/'+i)
    for i in glob_list:
        im = Image.open(i)
        preds = np.append(preds, model.predict((np.expand_dims(np.array(im.resize((299,299))), axis=0))/255), axis=1)
    negative_preds = preds.squeeze()
    negative_ground_truth = np.array([0 for i in preds.squeeze()])
    return positive_preds, positive_ground_truth, negative_preds, negative_ground_truth


def contiguous_true_vs_predicted(model, positive_class, negative_class, base_path, accepted_file_formats):
    """
    Simple wrapper for true_vs_predicted to get contiguous np arrays of true vs. predicted labels.
    :return: 2-tuple of np.arrays: contigous true values, contiguous predicted values
    """
    ppr, pgr, npr, ngr = true_vs_predicted(model, positive_class, negative_class, base_path, accepted_file_formats)
    return np.concatenate((pgr, ngr)), np.concatenate((ppr, npr))
