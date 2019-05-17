import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

from cnn_toolkit import Precision, Recall, filepattern, NeptuneMonitor, \
    pool_generator_classes, show_architecture, frosty, DebuggingCallback, \
    file_train_test_split, _variable_with_weight_decay, _variable_on_cpu, _activation_summary

from pathlib import Path

from keras import backend as K
from keras.layers import Dense, GlobalAveragePooling2D
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy

from tensornets import ResNet50v2


class CustomGenerator():
    """
    Custom generator compatible with the class-pooling function from cnn_toolkit
    """
    def __init__(self, df):
        self.df = df
        self.iterable_obj = df['x_col']
        self.counter = 0
        self.class_names = sorted(list(set(df['y_col'])))
        self.classes = [idx for idx, val in enumerate(self.class_names)]
        self.class_indices = {idx: val for idx, val in enumerate(self.class_names)}

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == len(self.iterable_obj):
            raise StopIteration
        item = self.iterable_obj[self.counter]
        self.counter += 1
        return item

    def __call__(self):
        return self.__next__(self)


class ImageHandler():
    def __init__(self, path_to_archive, img_shape):
        self.path_to_archive = path_to_archive
        self.img_shape = img_shape
        self.training_paths = pd.DataFrame()
        self.validation_paths = pd.DataFrame()

    def __call__(self):
        paths = file_train_test_split(self.path_to_archive, ['*.jpg', '*.jpeg', '*.png'])
        self.training_paths, self.validation_paths = paths

    def _load_single_image_as_tensor(self, idx):
        if type(idx) is str:
            idx = int(idx)
        img = tf.read_file(self.training_paths['x_col'][idx])  # read img
        label = tf.read_file(self.training_paths['y_col'][idx])
        img = tf.image.decode_image(img)  # decode to tf.Tensor
        return img, label

    def _preprocess_single_image(self, idx):
        img, label = self._load_single_image_as_tensor(idx)
        img = tf.image.resize_images(img, self.img_shape)  # resize to appropriate shape
        # above may require unsqueezing the tensor
        img = img/255.0  # rescale with float inference (necessary not to blow up the gradient at backprop)
        return img, label

    def _augment_single_image(self, idx):
        img, label = self._load_single_image_as_tensor(idx)
        img = tf.random_crop(img, img.shape)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        return img

    def train_generator(self):
        return CustomGenerator(self.training_paths)

    def validation_generator(self):
        return CustomGenerator(self.validation_paths)

    def show(self, idx):
        """
        Shows random or specified image from training set, post-preprocessing.
        Can show images from validation set when idx starts with 'v' letter.
        :param idx: string or integer index of a photo in
        a pd.DataFrame as returned by cnn_toolkit.file_train_test_split
        :return: None
        """
        if idx is not None:
            if idx.startswith('v'):
                num_idx = int(''.join(re.findall('(?!v).*', idx)))
                plt.imshow(self.validation_paths['x_col'][num_idx])
            else:
                num_idx = int(idx)
                plt.imshow(self.training_paths['x_col'][num_idx])
        else:
            idx = np.random.randint(0, self.training_paths.shape[0])
            plt.imshow(self.training_paths['x_col'][idx])


path_to_archive = Path.home() / Path('Downloads/FubarArchive/')
IMAGE_DIMS = (256, 256)
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 16
EPOCHS = 1

sess = tf.Session()  # create tensorflow session
K.set_session(sess)  # bind session to keras


imgl = ImageHandler(path_to_archive, IMAGE_DIMS)
imgl()  # call image loader to load paths

train = tf.data.Dataset.from_generator(
    imgl.train_generator(),
    output_types=tf.float64,
    output_shapes=IMAGE_DIMS
)

train = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

validation = tf.data.Dataset.from_generator(
    imgl.validation_generator(),
    output_types=tf.float64,
    output_shapes=IMAGE_DIMS
)

validation = validation.batch(BATCH_SIZE)

train_iter = train.make_one_shot_iterator()
val_iter = train.make_one_shot_iterator()


input_placeholder = tf.placeholder(tf.float32, shape=(None, *IMAGE_DIMS, 3))  # placeholder for inputs
y_true = tf.placeholder(tf.float32, shape=(None, 1))  # placeholder tensor for true labels

base = ResNet50v2(input_placeholder, is_training=True)

def architecture():
    # with tf.variable_scope('conv1') as scope:
    #     kernel = _variable_with_weight_decay('weights',
    #                                          shape=[5, 5, 3, 1024],
    #                                          # [filter_height * filter_width * in_channels, output_channels]
    #                                          stddev=5e-2,
    #                                          wd=None)
    #     conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
    #     biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.0))
    #     pre_activation = tf.nn.bias_add(conv, biases)
    #     conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #     _activation_summary(conv1)
    #
    # pool1
    pool_base = tf.nn.max_pool(base.output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool_base')
    # # norm1
    # norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
    #                   name='norm1')


    # output layer
    with tf.variable_scope('output_sigmoid') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.keras.layers.Flatten()(pool_base)
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 1],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [1], tf.constant_initializer(0.1))
        sigmoid_output = tf.nn.sigmoid(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(sigmoid_output)

    return sigmoid_output


y_pred = architecture()
loss = tf.reduce_mean(binary_crossentropy(y_true, y_pred))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)  # define optimizer and loss function
initializer = tf.global_variables_initializer()
sess.run(initializer)

with sess.as_default():  # make a training loop
    for i in range(EPOCHS+1):
        batch = train.next_batch(BATCH_SIZE)
        train_step.run(feed_dict={input_placeholder: batch[0],
                                  y_true: batch[1]})


recall_value = tf.metrics.recall(y_true, y_pred)
precision_value = tf.metrics.precision(y_true, y_pred)
acc_value = binary_accuracy(y_true, y_pred)
with sess.as_default():  # model evaluation
    batch = validation.next_batch(BATCH_SIZE)
    print(acc_value.eval(feed_dict={input_placeholder: batch[0],
                                    y_true: batch[1]}))
    print(recall_value.eval(feed_dict={input_placeholder: batch[0],
                                       y_true: batch[1]}))
    print(precision_value.eval(feed_dict={input_placeholder: batch[0],
                                          y_true: batch[1]}))

'''
# USE THIS TO MAKE SURE DATA IS LOADED FROM THE DRIVE
assert features.shape[0] == labels.shape[0]

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = ...
iterator = dataset.make_initializable_iterator()

sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})
'''







# graph = tf.Graph()
#
# with tf.Graph().as_default() as graph1:
#     input = tf.placeholder(tf.float32, (None, 20), name='input')
#     ...
#     output = tf.identity(input, name='output')
#
# with tf.Graph().as_default() as graph2:
#     input = tf.placeholder(tf.float32, (None, 20), name='input')
#     ...
#     output = tf.identity(input, name='output')
#
# graph = tf.get_default_graph()
# x = tf.placeholder(tf.float32, (None, 20), name='input')
