from cnn_toolkit import file_train_test_split

# ------------------------------
# YOLO-CONFIG FOR MASTER SCRIPT |
# ------------------------------

label_dict = {
    'lock': {
        'color': 1000
    },
    'rack': {
        'color': 2000
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

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------
# HERE LIVE THE IMAGES FOR CLASSIFIER TRAINING |
# ---------------------------------------------

file_formats = ['*.jpg', '*.jpeg', '*.png']
# path_to_archive = Path.home() / Path('fubar/FubarArchive/')
path_to_archive = '/home/ubuntu/darknet/AlexeyAB/darknet/result_img/'
paths = file_train_test_split(path_to_archive, file_formats, ignored_directories=['inference'])

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

