from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History

import PIL
import os
import numpy as np

import neptune as npt

from .cnn_toolkit import Precision, Recall, filepattern, NeptuneMonitor, pool_generator_classes

# -----------------------------
# OPTIONAL: INITIALIZE NEPTUNE |
# -----------------------------
# npt.init(api_token='insert_token_here',
#          project_qualified_name='user/fubar')
# npt.create_experiment(upload_source_files=[])  # keep what's inside parentheses to prevent neptune from reading code
# npt_monitor = NeptuneMonitor()

# -----------
# BASE MODEL |
# -----------
base = InceptionV3(weights='imagenet', include_top=False)
# ---------------------------------------------------------------------------------------------------------------------


# ----------------
# HYPERPARAMETERS |
# ----------------
INPUT_H = 280
INPUT_W = 280
BATCH_SIZE = 32
TRAIN_SIZE = 0
TEST_SIZE = 0
EPOCHS = 10
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------
# HERE LIVE THE IMAGES |
# ---------------------

path_to_archive = '/kjczarne/Downloads/FubarArchive/'

# ---------------------------------------------------------------------------------------------------------------------

# -------------------
# DATA PREPROCESSING |
# -------------------
training_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        validation_split=0.2)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Pool classes to exclude U-bar/LoopLock differentiation for the first model
class_pool_mapping = np.array([0, 0, 1, 1])
training_datagen= pool_generator_classes(training_datagen, class_pool_mapping)
validation_datagen = pool_generator_classes(validation_datagen, class_pool_mapping)

training_generator = training_datagen.flow_from_directory(
                path_to_archive,
                target_size=(INPUT_H, INPUT_W),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                subset='training')

validation_generator = validation_datagen.flow_from_directory(
                path_to_archive,
                target_size=(INPUT_H, INPUT_W),
                batch_size=BATCH_SIZE,
                class_mode='binary',
                subset='validation')


# ---------------------------------------------------------------------------------------------------------------------


# -------------------
# MODEL ARCHITECTURE |
# -------------------
y = base.output
y = GlobalAveragePooling2D()(y)
y = Dense(1024, activation='relu')(y)
y_pred = Dense(2, activation='sigmoid')(y)
# ---------------------------------------------------------------------------------------------------------------------


# -------------------------------
# CREATE MODEL FROM ARCHITECTURE |
# -------------------------------
model = Model(inputs=base.input, outputs=y_pred)
# ---------------------------------------------------------------------------------------------------------------------

# --------------
# FREEZE LAYERS |
# --------------
def freeze_layers(layer_iterable, view='len'):
    """
    freezes layers specified in the iterable passed to the function
    :param layer_iterable: an iterable of Keras layers with .trainable property
    :param view: switches return mode, can be 'len' or 'index'
    :return: returns number of frozen layers if view=='len', if view=='index' returns indices of all frozen layers
    """
    idx_record = []
    for idx, layer in enumerate(layer_iterable):
        assert layer.trainable is not None, "Item passed as layer has no property 'trainable'"
        layer.trainable = False
        idx_record.append(idx)

    if view == 'len':
        return "Number of frozen layers: {}".format(len(idx_record))
    elif view == 'index':
        return "Frozen layers: {}".format(idx_record)
    else:
        pass


freeze_layers(base.layers)  # this will freeze all base model layers
# ---------------------------------------------------------------------------------------------------------------------

# -----------------------------------
# COMPILE MODEL AND SET UP CALLBACKS |
# -----------------------------------
# always compile model after layers have been frozen

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', 'mae'])
precision = Precision()
recall = Recall()
history = History()
npt_monitor = NeptuneMonitor()
# ---------------------------------------------------------------------------------------------------------------------


# ----
# FIT |
# ----

model.fit_generator(training_generator,
                    steps_per_epoch=(TRAIN_SIZE / BATCH_SIZE),  # number of samples in the dataset
                    epochs=EPOCHS,  # number of epochs, training cycles
                    validation_data=validation_generator,  # performance eval on test set
                    validation_steps=(TEST_SIZE / BATCH_SIZE),
                    callbacks=[history, precision, recall, npt_monitor])
# read on SO, that the right way to compute precision and recall is to do it at the end of each epoch
# thus we use precision and recall functions as callbacks
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------
# RETRAINING THE MODEL |
# ---------------------
# todo - second training loop

# --------------------------------------
# EXPORT MODEL ARCHITECTURE AND WEIGHTS |
# --------------------------------------
# export model structure to json file:
model_struct_json = model.to_json()
filename = filepattern('model_ana_', '.json')
with open(filename, 'w') as f:
    f.write(model_struct_json)

# export weights to an hdf5 file:
w_filename = filepattern('weights_ana_', '.h5')
model.save_weights(w_filename)

# ---------------------------------------------------------------------------------------------------------------------

# ------------------------
# STOP NEPTUNE EXPERIMENT |
# ------------------------
# npt.stop()
