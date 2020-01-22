GPU_ID = "0"
#GPU ID(s) used for training, evaluation or testing. If you want use multiple GPUs print "0,1,2"

TRAIN_FLAG = False  # True if we want train new model or tune pretrained model from MODEL_PATH
                    # on data from TRAIN_PATH and VAL_PATH
TUNE_FLAG = False   # True if we want to tune pretrained model from MODEL_PATH on data from TRAIN_PATH and VAL_PATH
EVALUATION_FLAG = False # True if we want to evaluate pretrained model on data from VAL_PATH
#If TRAIN_FLAG == False and TUNE_FLAG == False and EVALUATION_FLAG == False
# prediction results of pretrained model from MODEL_PATH on TEST_PATH will be calculated and saved to RESULT_PATH

SAVE_DIR = 'models'
MODEL_NAME = 'unet_mct'

# dictionary with:
# "cateogory_name": [index, (R, G, B)]
# index is pixel intensity for segmentation categories (e.g. for deeplab format)
MASK_DICT = {
    "car": [1, (255, 255, 255)],
    "background": [0, (0, 0, 0)]
}

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


N_EPOCHS = 100
BATCH_SIZE = 8       #Batch size for training
EVAL_BATCH_SIZE = 1  #Batch size for validation, evaluation and testing

LEARNING_RATE = 1e-3  # Starting learning rate for default optimizer - Adam

# after the first EPOCHS_SCHEDULE_STEP formula for Learning rate update is used:
# NEW_LEARNING_RATE = LEARNING_RATE * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))
DROP_SCHEDULE_COEF = 0.8
EPOCHS_SCHEDULE_STEP = 5.0


TRAIN_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/train'
VAL_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/test'

#RESULT_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/results'
RESULT_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/raw_web_results'

#TEST_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/test/images'
TEST_PATH = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_dataset/raw_web'
# train_path = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_coco_dataset/train'
# val_path = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_coco_dataset/test'
#
# result_path = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_coco_dataset/results'
#
# test_path = '/media/cds-pc3/Data/pku_data_segmentation/pku_data_segmentation/car_segm_coco_dataset/test/images'

#MODEL_PATH = '/models/unet_light_mct2020-01-20-18-32-22.31-tloss-0.5139-tdice-0.7674-vdice-0.7584_car_segm.hdf5'
MODEL_PATH = '/models/unet_light_mct2020-01-21-12-31-11.14-tloss-0.5585-tdice-0.7510-vdice-0.7397_car_segm.hdf5'

# Dataset tree is
# TRAIN_PATH -- IMAGES_FOLDER_NAME
#            |- MASKS_FOLDER_NAME
#
# VAL_PATH -- IMAGES_FOLDER_NAME
#          |- MASKS_FOLDER_NAME
#

IMAGES_FOLDER_NAME = 'images'
MASKS_FOLDER_NAME = 'masks'

AUGMENT_CONFIG = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')