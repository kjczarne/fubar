from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

import matplotlib
matplotlib.use('TkAgg')

from cnn_toolkit import pool_generator_classes

from fubar_CONF import hprm, paths



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
