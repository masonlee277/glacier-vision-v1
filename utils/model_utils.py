# model_utils.py
# model_utils.py
from .common_imports import *
from .common_utils import *
from .data_utils import *
from .evaluation_utils import *
from .training_utils import *
from .image_utils import *
from .visualization_utils import *


def unetV1(sz=(512, 512, 1)):
    x = Input(sz)
    inputs = x
    f = 8
    layers = []
    for i in range(0, 6):
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        layers.append(x)
        x = MaxPooling2D()(x)
        f = f * 2
    ff2 = 64
    j = len(layers) - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j - 1
    for i in range(0, 5):
        ff2 = ff2 // 2
        f = f // 2
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same')(x)
        x = Concatenate(axis=3)([x, layers[j]])
        j = j - 1
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    x = Conv2D(f, 3, activation='relu', padding='same')(x)
    outputs = Conv2D(1, 1, activation='sigmoid')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(), loss=dice_loss, metrics=mean_iou)
    return model





def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_vgg16_unet(input_shape):
    """ Input """
    (w, h, c) = input_shape
    inputs = Input(input_shape)
    if c == 1:
        img_input = Input(shape=(w, h, 1))
        inputs = Concatenate()([img_input, img_input, img_input])
    ' Pre-trained VGG16 Model '
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    vgg16.trainable = False
    ' Encoder '
    s1 = vgg16.get_layer('block1_conv2').output
    s2 = vgg16.get_layer('block2_conv2').output
    s3 = vgg16.get_layer('block3_conv3').output
    s4 = vgg16.get_layer('block4_conv3').output
    ' Bridge '
    b1 = vgg16.get_layer('block5_conv3').output
    ' Decoder '
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    ' Output: Binary Segmentation '
    outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
    model = Model(inputs, outputs, name='VGG16_U-Net')
    return model

def compile_model(height: int, width: int, segmentation: bool=True) -> tf.keras.Model:
    """
  compile the model to be trained

  :param height: image height
  :param width: image width
  :param segmentation: bool flag to decide if to use binary classification or image segmentation. Default is segmentation
  :return: Compiled model
  """
    IMG_HEIGHT = height
    IMG_WIDTH = width
    IMG_CHANNELS = 3
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(input_shape)
    model_vgg16 = build_vgg16_unet(input_shape)
    model = model_vgg16
    loss = masked_dice_loss
    iou1 = tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5)
    iou2 = tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5)
    auc = tf.keras.metrics.AUC()
    aucLog = tf.keras.metrics.AUC(from_logits=True)
    metric = [mean_iou]
    opt = tf.keras.optimizers.legacy.Adam()
    model.compile(optimizer=opt, loss=loss, metrics=metric)
    print('total number of model parameters:', model.count_params())
    return model


def masked_loss(y_true, y_pred):
    """Defines a masked loss that ignores border/unlabeled pixels (represented as -1).

    Args:
      y_true: Ground truth tensor of shape [B, H, W, 1].
      y_pred: Prediction tensor of shape [B, H, W, N_CLASSES].
    """
    gt_validity_mask = tf.cast(tf.greater_equal(y_true[:, :, :, 0], 0), dtype=tf.float32)
    y_true = K.abs(y_true)
    raw_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    masked = gt_validity_mask * raw_loss
    return tf.reduce_mean(masked)


def masked_dice_loss(y_true, y_pred, mask, cont_loss=False, continuity_weight=0.15):
    smooth = 1.0
    y_true = tf.multiply(y_true, mask)
    y_pred = tf.multiply(y_pred, mask)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    if cont_loss:
        y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)
        line_length_penalty_scalar = line_length_penalty(y_pred_binary)
        total_loss = dice_loss + continuity_weight * line_length_penalty_scalar
    else:
        total_loss = dice_loss
    del y_true_f
    del y_pred_f
    del y_pred
    del y_true
    del mask
    return total_loss

def IoULoss(targets, inputs, smooth=1e-06):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = K.sum(K.dot(targets, inputs))
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return 1 - IoU


def s(targets, inputs, smooth=1e-06):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = BCE + dice_loss
    return Dice_BCE


def mean_iouV1(y_true, y_pred):
    yt0 = tf.cast(y_true[:, :, :, 0], 'float32')
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)), dtype='float32')
    union = tf.math.count_nonzero(tf.add(yt0, yp0), dtype='float32')
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))
    return iou

def mean_iou(y_true, y_pred):
    yt0 = y_true[:, :, :, 0]
    yp0 = K.cast(y_pred[:, :, :, 0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1.0, tf.cast(inter / union, 'float32'))
    return iou


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_lossV1(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.cast(tf.keras.backend.flatten(y_true), dtype=tf.float64)  # Cast to double tensor
    y_pred_f = tf.cast(tf.keras.backend.flatten(y_pred), dtype=tf.float64)  # Cast to double tensor
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss_with_weights(y_true, y_pred, w_tp=2.0, w_fp=0.01, w_tn=0.01, w_fn=1):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    tp = w_tp * tf.reduce_sum(y_true * y_pred)
    fp = w_fp * tf.reduce_sum((1 - y_true) * y_pred)
    tn = w_tn * tf.reduce_sum((1 - y_true) * (1 - y_pred))
    fn = w_fn * tf.reduce_sum(y_true * (1 - y_pred))
    dice_loss = 1 - (2 * tp + 1) / (2 * tp + fp + fn + 1)
    return dice_loss

def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    y_true_f = tf.cast(y_true_f, tf.float32)
    y_pred_f = tf.cast(y_pred_f, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_with_cross_entropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    dice = dice_loss(y_true, y_pred)
    y_true = tf.cast(y_true, tf.int32)
    print(y_true.dtype)
    ce = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight=tf.constant(2))
    return dice + ce


