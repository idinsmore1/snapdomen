from keras import backend as K

from keras.layers import (
    Input,
    Concatenate,
    Conv2D,
    ReLU,
    MaxPool2D,
    Conv2DTranspose,
    BatchNormalization
)

from keras.models import Model
from tensorflow.keras.optimizers import Adam


def encoder(input, n_filters, ksize, pool_size=(2, 2), pad='same'):
    # First Convolution
    conv1 = Conv2D(filters=n_filters, kernel_size=ksize, padding=pad)(input)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    # Second Convolution
    conv2 = Conv2D(filters=n_filters, kernel_size=ksize, padding=pad)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)

    # Concatenate
    concat = Concatenate()([conv2, input])
    # concat = conv2

    # Max Pooling Layer
    pooled = MaxPool2D(pool_size=pool_size)(concat)

    return concat, pooled


def decoder(input, con_block, n_filters, ksize, activation='relu', pad='same', stride=(2, 2)):
    # Upsample to increase tensor size
    upsample = Conv2DTranspose(filters=n_filters, kernel_size=(2, 2), strides=stride, padding=pad)(input)
    # Concatenate Results
    concat = Concatenate()([upsample, con_block])

    # First Convolution
    conv1 = Conv2D(filters=n_filters, kernel_size=ksize, padding=pad, activation=activation)(concat)
    conv1 = BatchNormalization()(conv1)

    # Second Convolution
    conv2 = Conv2D(filters=n_filters, kernel_size=ksize, padding=pad, activation=activation)(conv1)
    conv2 = BatchNormalization()(conv2)

    return conv2


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1.):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1 - pt_1), gamma)


def build_unet(input_shape, base_filter, ksize=(3, 3)):
    input_tensor = Input(input_shape)

    # Encoding section (Down slope of "U")
    down1, pool1 = encoder(input_tensor, base_filter, ksize)
    down2, pool2 = encoder(pool1, base_filter * 2, ksize)
    down3, pool3 = encoder(pool2, base_filter * 4, ksize)
    down4, pool4 = encoder(pool3, base_filter * 8, ksize)

    # Dilated Bottleneck
    dilated1 = Conv2D(base_filter * 16, ksize, activation='relu', padding='same', dilation_rate=1)(pool4)
    dilated2 = Conv2D(base_filter * 16, ksize, activation='relu', padding='same', dilation_rate=2)(dilated1)
    dilated3 = Conv2D(base_filter * 16, ksize, activation='relu', padding='same', dilation_rate=4)(dilated2)
    dilated4 = Conv2D(base_filter * 16, ksize, activation='relu', padding='same', dilation_rate=8)(dilated3)

    # Decoding Section (Up slope of "U")
    up4 = decoder(dilated4, down4, base_filter * 8, ksize)
    up3 = decoder(up4, down3, base_filter * 4, ksize)
    up2 = decoder(up3, down2, base_filter * 2, ksize)
    up1 = decoder(up2, down1, base_filter, ksize)

    out = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(up1)

    model = Model(inputs=[input_tensor], outputs=[out])
    model.compile(optimizer=Adam(learning_rate=5e-5), loss=focal_tversky, metrics=[dice_coef, tversky], run_eagerly=True)
    return model
