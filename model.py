import tensorflow
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *

from tensorflow.keras import backend as K


smooth = 1.

#special metrics for FCN training on small blobs - Dice
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef(y_true, y_pred, smooth, thresh):
    y_pred = y_pred > thresh
    y_pred = K.cast(y_pred, K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_coef_softmax(y_true, y_pred, smooth):
    # y_pred = y_pred == channel_id
    # y_pred = K.cast(y_pred, K.floatx())
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def iou_loss(smooth, thresh, channel_id):
  def iou_dice(y_true, y_pred):
    return 1.-iou_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  return iou_dice

def iou_metric(smooth, thresh, channel_id):
  def iou_channel(y_true, y_pred):
    return iou_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  iou_channel.__name__ = iou_channel.__name__ + "_" + str(channel_id)
  return iou_channel

def iou_metric_softmax(smooth, channel_id):
  def iou_channel(y_true, y_pred):

    y_pred_softmax = K.argmax(y_pred)
    y_pred_softmax = y_pred_softmax == channel_id
    y_pred_softmax = K.cast(y_pred_softmax, K.floatx())

    y_true_softmax = K.argmax(y_true)
    y_true_softmax = y_true_softmax == channel_id
    y_true_softmax = K.cast(y_true_softmax, K.floatx())

    return iou_coef_softmax(y_true_softmax, y_pred_softmax, smooth)
  iou_channel.__name__ = iou_channel.__name__ + "_" + str(channel_id)
  return iou_channel

def dice_metric(smooth, thresh, channel_id):
  def dice_channel(y_true, y_pred):
    return dice_coef(y_true[:,:,:,channel_id], y_pred[:,:,:,channel_id], smooth, thresh)
  dice_channel.__name__ = dice_channel.__name__ + "_" + str(channel_id)
  return dice_channel

#loss metrics for FCN training on base of Dice
def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

def dice_coef_multilabel(numLabels=13):
    def dice_coef_multi(y_true, y_pred):
        dice=0
        for index in range(numLabels):
            dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) # output tensor have shape (batch_size,
        return dice/numLabels                                           # width, height, numLabels)
    return dice_coef_multi

def dice_sparsed(sparse_thresh = 10):
    # useful for datasets with sparse object classes (only few images contains object class)
    # sparse_thresh - threshold for selection images with non zero number of pixels with object class
    # if number of pixels with object class (n_true_pixels) < sparse_thresh than we make dice equal to 0,
    # otherwise to common dice
    def dice_coef_sparsed(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        n_true_pixels = K.sum(y_true_f * y_true_f)
        dice_common = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
        dice_coef_sparsed_value = K.switch(n_true_pixels < sparse_thresh, 0.0, dice_common)
        return dice_coef_sparsed_value
    return dice_coef_sparsed


def categorical_crossentropy_loss(y_true, y_pred):
    cce = CategoricalCrossentropy()
    ccel = cce(y_true, y_pred)
    return ccel

def background_dice_loss(y_true, y_pred):
    bdl = dice_coef_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
    return bdl

def loss_function(y_true, y_pred):
    sum_loss =  categorical_crossentropy_loss(y_true, y_pred) + background_dice_loss(y_true, y_pred)
    return sum_loss

# def loss_function_1(y_true, y_pred):
#     cce = CategoricalCrossentropy()
#     ccel = cce(y_true, y_pred)
#     bdl = dice_coef_loss(y_true[:, :, :, 0], y_pred[:, :, :, 0])
#     sum_loss = ccel + bdl
#     return sum_loss

def dice_0(y_true, y_pred):
    return dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])

def unet_light_mct(pretrained_weights = None,input_size = (256,256,1),learning_rate = 1e-4, n_classes = 1, no_compile = False):
    inputs = Input(input_size)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv11 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv11)
    conv2 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool1)
    conv21 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv2)
    conv22 = Conv2D(32, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv21)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv22)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', dilation_rate=2, padding='same', kernel_initializer='he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)

    # up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop3))
    # Jason Brownlee. How to use the UpSampling2D and Conv2DTranspose Layers in Keras. 2019 https://machinelearningmastery.com/upsampling-and-transpose-convolution-layers-for-generative-adversarial-networks/
    # TensorRT Support Matrix. TensorRT 5.1.5. https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-support-matrix/index.html
    up8 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(drop3)
    merge8 = concatenate([conv22, up8], axis=3)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    # up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2))(conv8)
    merge9 = concatenate([conv11, up9], axis=3)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(n_classes, 1, activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    metrics = [dice_0]
    for channel_id in range(n_classes):
        metrics.append(iou_metric_softmax(smooth=1e-5, channel_id=channel_id))
    # iou_car_metric = iou_metric_softmax(smooth=1e-5, channel_id=1)
    # iou_background_metric = iou_metric_softmax(smooth=1e-5, channel_id=0)

    if no_compile == False:
        model.compile(optimizer=Adam(lr=learning_rate),
                      run_eagerly = True,
                      loss=loss_function,
                      metrics=metrics)

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model