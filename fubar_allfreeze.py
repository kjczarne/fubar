from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import History

import matplotlib
matplotlib.use('TkAgg')

import neptune as npt
import tensorflow as tf

from cnn_toolkit import filepattern, NeptuneMonitor, show_architecture, frosty, make_pred_output_callback
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


# -------------------
# MODEL ARCHITECTURE |
# -------------------
x = tf.keras.layers.Input(shape=(hprm['INPUT_H'], hprm['INPUT_W'], 3))
                          # batch_size=hprm['BATCH_SIZE'])
x = tf.keras.layers.Dropout(rate=0.2)(x)
base = tf.keras.applications.InceptionV3(input_tensor=x, weights='imagenet', include_top=False)
y = base.output
y = tf.keras.layers.GlobalAveragePooling2D()(y)  # __call__()
y = tf.keras.layers.Dense(1024, activation='relu', name='my_dense_1024')(y)
y_pred = tf.keras.layers.Dense(1, activation='sigmoid', name='output_dense')(y)
# ---------------------------------------------------------------------------------------------------------------------


# -------------------------------
# CREATE MODEL FROM ARCHITECTURE |
# -------------------------------
model = Model(inputs=base.input, outputs=y_pred)
# tf.Tensor objects are associated with tf.Graph object, which stores the architecture of the model,
# that's why for functional Keras Model object all we need to do is specify the input and output tensor
# and Model object figures the actual architecture from tf.Graph
# ---------------------------------------------------------------------------------------------------------------------

# --------------
# FREEZE LAYERS |
# --------------
frosty(base.layers)  # this will freeze all base model layers
# ---------------------------------------------------------------------------------------------------------------------

# -----------------------------------
# COMPILE MODEL AND SET UP CALLBACKS |
# -----------------------------------
# always compile model AFTER layers have been frozen
recall = tf.keras.metrics.Recall()
precision = tf.keras.metrics.Precision()
validation_output_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=make_pred_output_callback(
    model,
    validation_generator,
    hprm['BATCH_SIZE']))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',
                                                                        recall,
                                                                        precision])

history = History()
npt_monitor = NeptuneMonitor(hprm['BATCH_SIZE'])
# ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------
# TRAIN TOP LAYERS ON NEW DATA FOR A FEW EPOCHS|
# ---------------------------------------------
post_training_model = model.fit_generator(training_generator,
                                          steps_per_epoch=((hprm['TRAIN_SIZE'] // hprm['BATCH_SIZE'])+1),
                                          epochs=10,  # number of epochs, training cycles
                                          validation_data=validation_generator,  # performance eval on test set
                                          validation_steps=((hprm['TEST_SIZE'] // hprm['BATCH_SIZE'])+1),
                                          verbose=1,
                                          callbacks=[history,
                                                     npt_monitor])
                                                     # validation_output_callback])

y_pred = model.predict_generator(validation_generator,
                                 steps=(hprm['TEST_SIZE'] // hprm['BATCH_SIZE'])+1,
                                 callbacks=[],
                                 verbose=1)
# ---------------------------------------------------------------------------------------------------------------------

# --------------------------------------
# EXPORT MODEL ARCHITECTURE AND WEIGHTS |
# --------------------------------------
# export model structure to json file:
model_struct_json = model.to_json()
filename = filepattern('model_allfreeze', '.json')
with open(filename, 'w') as f:
    f.write(model_struct_json)

# export weights to an hdf5 file:
w_filename = filepattern('weights_allfreeze', '.h5')
model.save_weights(w_filename)
# ---------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------
# VISUALIZE BASE ARCHITECTURE TO DECIDE WHICH LAYERS TO FREEZE |
# -------------------------------------------------------------
# PUT BREAKPOINT HERE!!!!!!!!!!!!!!!
print(list(show_architecture(base)))
# INSERT DEBUGGER BREAKPOINT DIRECTLY ON THE NEXT COMMAND TO VIEW THE ARCHITECTURE AT RUNTIME
# ---------------------------------------------------------------------------------------------------------------------

# ------------------------
# STOP NEPTUNE EXPERIMENT |
# ------------------------
npt.stop()

# ======================================================================================================================
# ======================================================================================================================

# # --------------
# # FREEZE LAYERS |
# # --------------
# # for now I just pass a slice of layers used in Keras documentation
# frosty(model.layers[:249], frost=True)
# frosty(model.layers[249:], frost=False)
#
# # -----------------------------
# # OPTIONAL: INITIALIZE NEPTUNE |
# # -----------------------------
# npt.init(api_token=npt_token,
#          project_qualified_name=npt_project)
# npt.create_experiment(upload_source_files=[])  # keep what's inside parentheses to prevent neptune from reading code
# npt_monitor = NeptuneMonitor(BATCH_SIZE)
#
#
# # ------------------------------------
# # COMPILE MODEL AGAIN AND TRAIN AGAIN |
# # ------------------------------------
# # always compile model AFTER layers have been frozen
# recall = tf.keras.metrics.Recall()
# precision = tf.keras.metrics.Precision()
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                   min_delta=0.001,
#                                                   patience=0,
#                                                   verbose=0,
#                                                   mode='auto',
#                                                   baseline=None,
#                                                   restore_best_weights=True)
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc',
#                                                                         recall,
#                                                                         precision])
# post_training_model = model.fit_generator(training_generator,
#                                           steps_per_epoch=(TRAIN_SIZE / BATCH_SIZE),  # number of samples in the dataset
#                                           epochs=EPOCHS,  # number of epochs, training cycles
#                                           validation_data=validation_generator,  # performance eval on test set
#                                           validation_steps=(TEST_SIZE / BATCH_SIZE),
#                                           callbacks=[history,
#                                                      npt_monitor,
#                                                      early_stopping])
#
# y_pred = model.predict_generator(validation_generator,
#                                  steps=(TEST_SIZE // BATCH_SIZE)+1,
#                                  callbacks=[],
#                                  verbose=1)
#
# # --------------------------------------
# # EXPORT MODEL ARCHITECTURE AND WEIGHTS |
# # --------------------------------------
# # export model structure to json file:
# model_struct_json = model.to_json()
# filename = filepattern('model_partfreeze', '.json')
# with open(filename, 'w') as f:
#     f.write(model_struct_json)
#
# # export weights to an hdf5 file:
# w_filename = filepattern('weights_partfreeze', '.h5')
# model.save_weights(w_filename)
#
# # ---------------------------------------------------------------------------------------------------------------------
#
# # ------------------------
# # STOP NEPTUNE EXPERIMENT |
# # ------------------------
# npt.stop()
# # ======================================================================================================================
# # ======================================================================================================================
