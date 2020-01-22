# itl-deep-seg
Scripts for deep segmentation with custom losses, metrics, light models, pre- and post-processing with Tensorflow 2 (keras-based)

## Segmentation and auxiliary Python-scripts

**train_test.py** - training and testing of segmentation model in single python-script

              You should just run this script for training, evaluation or testing deep neural 
              network model. All main configs and mode selection in config.py 
              

**config.py** - all main configs and settings for scripts

              TRAIN_FLAG = True if we want to train new model or tune pretrained model from MODEL_PATH
                                 on data from TRAIN_PATH and VAL_PATH
                                 
              TUNE_FLAG = True if we want to tune pretrained model from MODEL_PATH on data from TRAIN_PATH and VAL_PATH
              
              EVALUATION_FLAG = True if we want to evaluate pretrained model on data from VAL_PATH
              
              If TRAIN_FLAG == False and TUNE_FLAG == False and EVALUATION_FLAG == False
              prediction results of pretrained model from MODEL_PATH on TEST_PATH will be calculated and saved to RESULT_PATH
             
              MASK_DICT - dictionary with indexes and color pallete for object categories (classes):
              {              
                  "cateogory_name": [index, (R, G, B)],
                  ...                  
              }
              index is pixel intensity for segmentation categories (e.g. for deeplab format)

**data_preprocess.py** - data generators, dataset preparation, data post processing

              Features:                  
                  - Conversion grayscale masks with class indexes to color mask with pallete according to mask_dict                             - Training generator based on color masks with pallete

**model.py** - different metrics, losses and segmentation model

              Implemented models: 
                  - UNetMCT (Light UNet-like architecture with ConvTranspose layers)
                  
              Implemented metrics:                   
                  - Dice                  
                  - Sparsed Dice                  
                  - IoU (Intersection over Union)
                  
              Implemented losses:                   
                  - Dice loss                  
                  - Multiclass Dise loss                  
                  - Sparsed Dice Loss                  
                  - Categorical Crossentropy Loss                  
                  - Mixed Loss Function

## Folders in the repository:

**models** - pretrained models

## Examples

**Car segmentation in bounding box (on test images)**

![UNetMCT results on test images](/assets/Results_on_test.jpg)
