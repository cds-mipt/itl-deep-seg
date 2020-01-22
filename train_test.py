from model import *
from data_processing import *
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import time

from config import *

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
train_flag = TRAIN_FLAG
tune_flag = TUNE_FLAG
evaluation_flag = EVALUATION_FLAG

save_dir = SAVE_DIR
model_name = MODEL_NAME

# dictionary with:
# "cateogory_name": [index, (R, G, B)]
# index is pixel intensity for segmentation categories (e.g. for deeplab format)
mask_dict = MASK_DICT

image_width = IMAGE_WIDTH
image_height = IMAGE_HEIGHT


n_epochs = N_EPOCHS
batch_size = BATCH_SIZE
eval_batch_size = EVAL_BATCH_SIZE

learning_rate = LEARNING_RATE

drop_schedule_coef = DROP_SCHEDULE_COEF
epochs_schedule_step = EPOCHS_SCHEDULE_STEP

train_path = TRAIN_PATH
val_path = VAL_PATH

result_path = RESULT_PATH

test_path = TEST_PATH

current_dir = os.getcwd()
project_dir = os.path.dirname(current_dir)
model_path = current_dir+MODEL_PATH


train_sample_len = len(os.listdir(train_path+'/'+IMAGES_FOLDER_NAME))
test_sample_len = len(os.listdir(val_path+'/'+IMAGES_FOLDER_NAME))

train_steps = train_sample_len//batch_size
test_steps = test_sample_len//eval_batch_size


os.makedirs(current_dir+'/logs',exist_ok=True)
timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
log_filename = current_dir+'/logs/log-'+model_name+'-'+ timestr +'.txt'
with open(log_filename, "a") as log_file:
    log_file.write("Experiment start: "+ log_filename + "\n")
    log_file.write("Input dataset(s):\n")
    log_file.write("Train path: "+ train_path +"\n")
    log_file.write("Val path: "+ val_path +"\n")


data_gen_args = AUGMENT_CONFIG

with open(log_filename, "a") as log_file:
    log_file.write("Augmentation args: "+ str(data_gen_args) + "\n")


class_items = mask_dict.items()
num_class = len(class_items)


if not tune_flag:
    if train_flag:
        model_path = None

with open(log_filename, "a") as log_file:
    log_file.write("Model path: "+ str(model_path) + "\n")

    model = unet_light_mct(pretrained_weights=model_path, input_size=(image_height, image_width, 3),
                       learning_rate=learning_rate, n_classes=num_class, no_compile=False)
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)



    with open(log_filename, "a") as log_file:
        log_file.write('\nModel summary:\n')
        log_file.write(short_model_summary)
        log_file.write('Metrics:' + str(model.metrics) + '\n')
        log_file.write('Image_Size: ' + str(image_width) + ' x '+str(image_height) + '\n')
        log_file.write('Batch_Size: ' + str(batch_size) + '\n')
        log_file.write('\nTraining process:\n')
        log_file.write('\nEpoch no, train dice, train loss, val dice, val loss\n')
    if train_flag:
        os.makedirs(current_dir + '/'+ save_dir, exist_ok=True)
        train_gen = train_generator(batch_size=batch_size,
                                    train_path=train_path,
                                    image_folder=IMAGES_FOLDER_NAME,
                                    mask_folder=MASKS_FOLDER_NAME,
                                    aug_dict=data_gen_args,
                                    image_color_mode='rgb',
                                    mask_color_mode="rgb",
                                    flag_multi_class=True,
                                    num_class=num_class,
                                    target_size=(image_height, image_width),
                                    save_to_dir=None,
                                    mask_dict=mask_dict)
        val_gen = train_generator(batch_size=eval_batch_size,
                                  train_path=val_path,
                                  image_folder=IMAGES_FOLDER_NAME,
                                  mask_folder=MASKS_FOLDER_NAME,
                                  aug_dict=dict(),
                                  image_color_mode='rgb',
                                  mask_color_mode="rgb",
                                  flag_multi_class=True,
                                  num_class=num_class,
                                  target_size=(image_height, image_width),
                                  save_to_dir=None,
                                  mask_dict=mask_dict
                                  )


        model_checkpoint = ModelCheckpoint(current_dir + '/' +save_dir+ '/'+ model_name + timestr +
                                           '.{epoch:02d}-tloss-{loss:.4f}-tdice-{dice_0:.4f}-vdice-{val_dice_0:.4f}.hdf5',
                                           monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, save_format='h5')

        def scheduler(epoch, learning_rate):
            if epoch < epochs_schedule_step:
                return learning_rate
            else:
                drop = drop_schedule_coef
                epochs_drop = epochs_schedule_step
                lrate = learning_rate * tf.math.pow(drop,
                                                 tf.math.floor((1 + epoch) / epochs_drop))
                return lrate


        callback_LearningRateScheduler = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)

        callbacks = [model_checkpoint,
                     callbacks.CSVLogger(log_filename, separator=',', append=True),
                     tensorflow.keras.callbacks.TensorBoard(log_dir='./logs'),
                     callback_LearningRateScheduler
                     ]

        start_time = time.time()

        model.fit_generator(generator=train_gen,
                            steps_per_epoch=train_steps,
                            epochs=n_epochs,
                            validation_data=val_gen,
                            validation_steps=test_sample_len,
                            callbacks=callbacks)

        end_time = time.time()
        duration = end_time - start_time
        with open(log_filename, "a") as log_file:
            log_file.write("Training time, sec: "+ str(duration) + "\n")
    else:
        if evaluation_flag:

            eval_gen = train_generator(batch_size=eval_batch_size,
                                      train_path=val_path,
                                      image_folder=IMAGES_FOLDER_NAME,
                                      mask_folder=MASKS_FOLDER_NAME,
                                      aug_dict=dict(),
                                      image_color_mode='rgb',
                                      mask_color_mode="rgb",
                                      flag_multi_class=True,
                                      num_class=num_class,
                                      target_size=(image_height, image_width),
                                      save_to_dir=None,
                                      mask_dict=mask_dict
                                      )
            model.evaluate_generator(generator=eval_gen,
                                steps=test_steps,
                                verbose=1)

        else:
            # Run when train_flag == False, tune_flag == False, evaluation_flag == False
            os.makedirs(result_path, exist_ok=True)

            # test_gen = test_generator(test_path=test_path,
            #                          num_image = test_sample_len,
            #                          target_size = (image_height, image_width))

            for count, filename in enumerate(sorted(os.listdir(test_path))):
                img, raw, scale_coef = read_and_preprocess_image(test_path=test_path,
                                                                 filename=filename,
                                                                 target_size=(image_height, image_width))
                start_time = time.time()
                results = model.predict(img, verbose=1)
                end_time = time.time()
                duration = end_time - start_time

                out_file_name = os.path.join(result_path, filename.split('.')[0] + '.png')
                img = label_visualize_precise(mask_dict=mask_dict,
                                      img=results[0],
                                      scale_coef = scale_coef)
                io.imsave(out_file_name, img)
                print(str(count)+':' + os.path.join(test_path, filename))
