from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing import image
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from keras import metrics
import PIL


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
# ---------------------------------------------------------------------------------------------------------------------


# -------------------|
# DATA PREPROCESSING |
# -------------------|
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

# ---------------
# CUSTOM METRICS |
# ---------------
# at least in the beginning we start out with a very imbalanced dataset, thus we need to implement additional metrics


def precision(y_true, y_pred):  # this is the order of arguments recognized by keras in metrics kwarg
                                # in the model compiler
    raise RuntimeError('Not implemented yet!')


def recall(y_true, y_pred):
    raise RuntimeError('Not implemented yet!')

# --------------
# COMPILE MODEL |
# --------------
# always compile model after layers have been frozen

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc', 'mae', precision, recall])


# ----
# FIT |
# ----
