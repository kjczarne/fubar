from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('TkAgg')

from pathlib import Path

from cnn_toolkit import filepattern, NeptuneMonitor, \
    pool_generator_classes, show_architecture, frosty, \
    file_train_test_split

# ---------------------
# HERE LIVE THE IMAGES |
# ---------------------

file_formats = ['*.jpg', '*.jpeg', '*.png']
# path_to_archive = Path.home() / Path('fubar/FubarArchive/')
path_to_archive = '/home/ubuntu/darknet/AlexeyAB/darknet/result_img/'
paths = file_train_test_split(path_to_archive, file_formats, ignored_directories=['inference'])

# ---------------------------------------------------------------------------------------------------------------------

# ----------------
# HYPERPARAMETERS |
# ----------------
hprm = dict(
    INPUT_H=299,
    INPUT_W=299,
    BATCH_SIZE=32,
    TRAIN_SIZE=paths[0].shape[0],
    TEST_SIZE=paths[1].shape[0],
    EPOCHS=10,
    EARLY_STOPPING_DELTA=0.001
)
# ---------------------------------------------------------------------------------------------------------------------

# -------------------
# DATA PREPROCESSING |
# -------------------
# We need a random split of 80/20 for training and validation images. Make a DF mapping files from random categories
# to a validation or training set and use it to construct training_generator and validation_generator

test_image_datagen = ImageDataGenerator(
        rescale=1./255)

training_image_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True)

training_generator = training_image_datagen.flow_from_dataframe(
                paths[0],
                x_col='x_col',
                y_col='y_col',
                target_size=(hprm['INPUT_H'], hprm['INPUT_W']),
                batch_size=hprm['BATCH_SIZE'],
                class_mode='sparse')

validation_generator = test_image_datagen.flow_from_dataframe(
                paths[1],
                x_col='x_col',
                y_col='y_col',
                target_size=(hprm['INPUT_H'], hprm['INPUT_W']),
                batch_size=hprm['BATCH_SIZE'],
                class_mode='sparse')

# Pool classes to exclude U-bar/LoopLock differentiation for the first model
# class_pool_mapping = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}
# class_pool_mapping = {0: 0, 1: 0, 2: 1, 3: 1}
class_pool_mapping = {0: 0, 1: 1}
pool_generator_classes(training_generator, class_pool_mapping)
pool_generator_classes(validation_generator, class_pool_mapping)
validation_generator.class_mode = 'binary'  # bring class mode back to binary
training_generator.class_mode = 'binary'

# ---------------------------------------------------------------------------------------------------------------------
