from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import History

import matplotlib
matplotlib.use('TkAgg')

from pathlib import Path

import neptune as npt
import tensorflow as tf
import json

from cnn_toolkit import filepattern, NeptuneMonitor, \
    pool_generator_classes, show_architecture, frosty, \
    file_train_test_split
from fubar_preprocessing import hprm, training_generator, validation_generator

from npt_token_file import project_path, api
npt_token = api
npt_project = project_path

# -----------------------------
# OPTIONAL: INITIALIZE NEPTUNE |
# -----------------------------
npt.init(api_token=npt_token,
         project_qualified_name=npt_project)
npt.create_experiment(upload_source_files=[])  # keep what's inside parentheses to prevent neptune from reading code

# ----------------
# HYPERPARAMETERS |
# ----------------
print(hprm)
# ---------------------------------------------------------------------------------------------------------------------


# ----------------
# RE-CREATE MODEL |
# ----------------
with open(input('Enter filename with model architecture: '), 'r') as f:
    model = tf.keras.models.model_from_json(f.read())
model.load_weights(input('Enter filename with model weights: '))
# ---------------------------------------------------------------------------------------------------------------------

history = History()
npt_monitor = NeptuneMonitor(hprm['BATCH_SIZE'])
# ---------------------------------------------------------------------------------------------------------------------

# --------------
# FREEZE LAYERS |
# --------------
# for now I just pass a slice of layers used in Keras documentation
frosty(model.layers[:249], frost=True)
frosty(model.layers[249:], frost=False)

# -----------------------------
# OPTIONAL: INITIALIZE NEPTUNE |
# -----------------------------
npt.init(api_token=npt_token,
         project_qualified_name=npt_project)
npt.create_experiment(upload_source_files=[])  # keep what's inside parentheses to prevent neptune from reading code
npt_monitor = NeptuneMonitor(hprm['BATCH_SIZE'])


# ------------------------------------
# COMPILE MODEL AGAIN AND TRAIN AGAIN |
# ------------------------------------
# always compile model AFTER layers have been frozen
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  min_delta=hprm['EARLY_STOPPING_DELTA'],
                                                  patience=0,
                                                  verbose=0,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=True)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',
                                                                        recall,
                                                                        precision])
post_training_model = model.fit_generator(training_generator,
                                          steps_per_epoch=(hprm['TRAIN_SIZE'] / hprm['BATCH_SIZE']),  # number of samples in the dataset
                                          epochs=hprm['EPOCHS'],  # number of epochs, training cycles
                                          validation_data=validation_generator,  # performance eval on test set
                                          validation_steps=(hprm['TEST_SIZE'] / hprm['BATCH_SIZE']),
                                          callbacks=[history,
                                                     npt_monitor,
                                                     early_stopping])

y_pred = model.predict_generator(validation_generator,
                                 steps=(hprm['TEST_SIZE'] // hprm['BATCH_SIZE'])+1,
                                 callbacks=[],
                                 verbose=1)

# --------------------------------------
# EXPORT MODEL ARCHITECTURE AND WEIGHTS |
# --------------------------------------
# export model structure to json file:
model_struct_json = model.to_json()
filename = filepattern('model_partfreeze', '.json')
with open(filename, 'w') as f:
    f.write(model_struct_json)

# export weights to an hdf5 file:
w_filename = filepattern('weights_partfreeze', '.h5')
model.save_weights(w_filename)

# ---------------------------------------------------------------------------------------------------------------------

# ------------------------
# STOP NEPTUNE EXPERIMENT |
# ------------------------
npt.stop()
# ======================================================================================================================
# ======================================================================================================================
