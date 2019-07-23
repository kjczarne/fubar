from cnn_toolkit import file_train_test_split, filepattern, get_fresh_weights_and_model
import pandas as pd
import os

# ------------------------------
# YOLO-CONFIG FOR MASTER SCRIPT |
# ------------------------------
# colors are in BGR format consistent with CV2
label_dict = {
    'lock': {
        'color': (226, 4, 185)
    },
    'rack': {
        'color': (67, 226, 4)
    }
}

labels_of_images_to_be_cropped = 'lock'

tf_s_conf = dict(
    host='localhost',
    port='8501',
    model_name='fubar',
    model_version='4',
    batch_size=1,
    signature_name='serving_default'
)

ignore_existing_train_test_split = False
# ---------------------------------------------------------------------------------------------------------------------

# ----------------------
# MASTER PATH CONF DICT |
# ----------------------
path_conf = {
    'yolo_cfg': '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/cfg/yolo-obj.cfg',
    'yolo_weights': '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/backup/yolo-obj_final.weights',
    'yolo_darknet_app': '/home/ubuntu/darknet/AlexeyAB/darknet/',
    'yolo_test_set': '/home/ubuntu/darknet/images/test',
    'yolo_training_set': '/home/ubuntu/darknet/images/train',
    'yolo_obj.data': '/home/ubuntu/darknet/AlexeyAB/darknet/build/darknet/x64/data/obj.data',
    'cropped_images_for_training': '/home/ubuntu/darknet/AlexeyAB/darknet/result_img/'
}

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------
# HERE LIVE THE IMAGES FOR CLASSIFIER TRAINING |
# ---------------------------------------------

file_formats = ['*.jpg', '*.jpeg', '*.png']
path_to_archive = path_conf['cropped_images_for_training']
try:
    tr, t = get_fresh_weights_and_model(os.getcwd(), 'train*.csv', 'test*.csv')
except IndexError:
    t = []
if len(t) == 0 or ignore_existing_train_test_split:
    paths = file_train_test_split(path_to_archive, file_formats, ignored_directories=['inference'])
    # we use default 80/20 split
else:
    paths = pd.read_csv(tr), pd.read_csv(t)
    # but if any train-test split files already exist, we reuse them
    # this way fubar_unfreeze won't perform another split before training but will reuse same val. data

# ---------------------------------------------------------------------------------------------------------------------

# -----------------------------------
# HYPERPARAMETERS FOR THE CLASSIFIER |
# -----------------------------------


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
