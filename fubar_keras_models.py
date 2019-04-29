from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics

from IPython.display import display

import PIL
import neptune as npt
import os

from .cnn_toolkit import Precision, Recall, filepattern

# -----------------------------
# OPTIONAL: INITIALIZE NEPTUNE |
# -----------------------------
# npt.init(api_token='insert_token_here',
#          project_qualified_name='user/fubar')
# npt.create_experiment(upload_source_files=[])  # keep what's inside parentheses to prevent neptune from reading code

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


# ---------------------------------------------------------------------------------------------------------------------

# -------------------
# DATA PREPROCESSING |
# -------------------
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True)

training_generator = train_datagen.flow_from_directory(
                'training',
                target_size=(INPUT_H, INPUT_W),
                batch_size=BATCH_SIZE,
                class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
                'test',
                target_size=(INPUT_H, INPUT_W),
                batch_size=BATCH_SIZE,
                class_mode='binary')
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

# --------------
# COMPILE MODEL |
# --------------
# always compile model after layers have been frozen

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', 'mae'])
precision = Precision()
recall = Recall()
# ---------------------------------------------------------------------------------------------------------------------


# ----
# FIT |
# ----

model.fit_generator(train_generator,
                    steps_per_epoch=(TRAIN_SIZE / BATCH_SIZE),  # number of samples in the dataset
                    epochs=EPOCHS,  # number of epochs, training cycles
                    validation_data=validation_generator,  # performance eval on test set
                    validation_steps=(TEST_SIZE / BATCH_SIZE),
                    callbacks=[history, precision, recall])
# read on SO, that the right way to compute precision and recall is to do it at the end of each epoch
# thus we use precision and recall functions as callbacks
# ---------------------------------------------------------------------------------------------------------------------

# ------------------------
# SEND METRICS TO NEPTUNE |
# ------------------------
# npt.send_metric('acc', 0.95)
# npt.send_metric('mae', 0.95)
# npt.send_metric(precision, 0.95)
# npt.send_metric(recall, 0.95)
# npt.stop()
