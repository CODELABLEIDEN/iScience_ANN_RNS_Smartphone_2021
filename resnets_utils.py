from tensorflow.keras.layers import ZeroPadding2D, Input, MaxPooling2D, AveragePooling2D, Add
from tensorflow.keras.models import Model

from MDN import *

from tensorflow.keras.layers import Dense, Bidirectional, Activation, Concatenate, GaussianNoise, SpatialDropout2D

# MOST OF THIS CODE originally belongs to https://github.com/priya-dwivedi/Deep-Learning/tree/master/resnet_keras

def identity_block(X, f, filters, stage, block, drop=0.0, reg2=0.0):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = TimeDistributed(
        Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2a'))(X)
    X = TimeDistributed(Activation('relu'))(X)

    # X = TimeDistributed(SpatialDropout2D(rate=drop))(X)

    # Second component of main path (≈3 lines)
    X = TimeDistributed(
        Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2b'))(X)
    X = TimeDistributed(Activation('relu'))(X)

    # X = TimeDistributed(SpatialDropout2D(rate=drop))(X)

    # Third component of main path (≈2 lines)
    X = TimeDistributed(
        Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2c'))(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = TimeDistributed(Activation('relu'))(X)

    # X = TimeDistributed(SpatialDropout2D(rate=drop))(X)

    return X


# %%

def convolutional_block(X, f, filters, stage, block, s=2, reg2=0.0):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # MAIN PATH
    # First component of main path
    X = TimeDistributed(
        Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2a'))(X)
    X = TimeDistributed(Activation('relu'))(X)

    # Second component of main path (≈3 lines)
    X = TimeDistributed(
        Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2b'))(X)
    X = TimeDistributed(Activation('relu'))(X)

    # Third component of main path (≈2 lines)
    X = TimeDistributed(
        Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '2c'))(X)

    # SHORTCUT PATH #### (≈2 lines)
    X_shortcut = TimeDistributed(
        Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X_shortcut)
    X_shortcut = TimeDistributed(BatchNormalization(axis=3, name=bn_name_base + '1'))(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = TimeDistributed(Activation('relu'))(X)

    return X


def TDResNet50(input_shape=(9, 100, 100, 3), components=3, reg=0., num_detectors=1,
               factor=1, gauss_noise=False, reg2=0.):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = TimeDistributed(ZeroPadding2D((3, 3)))(X_input)

    # Stage 1
    X = TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2), name='conv1',
                               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(reg2)))(X)
    X = TimeDistributed(BatchNormalization(axis=3, name='bn_conv1'))(X)
    X = TimeDistributed(Activation('relu'))(X)
    X = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2)))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[32 / factor, 32 / factor, 128 / factor], stage=2, block='a', s=1)
    X = identity_block(X, 3, [32 / factor, 32 / factor, 128 / factor], stage=2, block='b')
    X = identity_block(X, 3, [32 / factor, 32 / factor, 128 / factor], stage=2, block='c')

    # Stage 3 (≈4 lines)
    X = convolutional_block(X, f=3, filters=[64 / factor, 64 / factor, 256 / factor], stage=3, block='a', s=2)
    X = identity_block(X, 3, [64 / factor, 64 / factor, 256 / factor], stage=3, block='b')
    X = identity_block(X, 3, [64 / factor, 64 / factor, 256 / factor], stage=3, block='c')
    X = identity_block(X, 3, [64 / factor, 64 / factor, 256 / factor], stage=3, block='d')

    # Stage 4 (≈6 lines)
    X = convolutional_block(X, f=3, filters=[128 / factor, 128 / factor, 512 / factor], stage=4, block='a', s=2)
    X = identity_block(X, 3, [128 / factor, 128 / factor, 512 / factor], stage=4, block='b')
    X = identity_block(X, 3, [128 / factor, 128 / factor, 512 / factor], stage=4, block='c')
    X = identity_block(X, 3, [128 / factor, 128 / factor, 512 / factor], stage=4, block='d')
    X = identity_block(X, 3, [128 / factor, 128 / factor, 512 / factor], stage=4, block='e')
    X = identity_block(X, 3, [128 / factor, 128 / factor, 512 / factor], stage=4, block='f')

    # Stage 5 (≈3 lines)
    X = convolutional_block(X, f=3, filters=[256 / factor, 256 / factor, 1024 / factor], stage=5, block='a', s=2)
    X = identity_block(X, 3, [256 / factor, 256 / factor, 1024 / factor], stage=5, block='b')
    X = identity_block(X, 3, [256 / factor, 256 / factor, 1024 / factor], stage=5, block='c')

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    X = TimeDistributed(AveragePooling2D((2, 2), name="avg_pool"))(X)

    # output layer
    X = TimeDistributed(Flatten())(X)  # (None, 9, )
    X = LSTM(50, return_sequences=True, dropout=0.4)(X)
    X = LSTM(50, return_sequences=True, dropout=0.4)(X)
    X = LSTM(50, return_sequences=False)(X)

    outs = []

    for i in range(num_detectors):
        # X = Dense(256, activation='relu', name=f"pre1_{i}", kernel_regularizer=l1(reg))(X)
        # X = Dense(512, activation='relu', name=f"pre2_{i}", kernel_regularizer=l1(reg))(X)
        alphas = Dense(components, activation="softmax", name=f"alphas_{i}", kernel_regularizer=l1(reg))(X)
        mus = Dense(components, name=f"mus_{i}", kernel_regularizer=l1(reg))(X)
        sigmas = Dense(components, activation="nnelu", name=f"sigmas_{i}", kernel_regularizer=l1(reg))(X)

        outs.append(Concatenate(name=f"pvec_{i}")([alphas, mus, sigmas]))

    # Create model
    model = Model(inputs=X_input, outputs=outs, name='ResNet50')

    return model


if __name__ == '__main__':
    with tf.device('/GPU:1'):
        model = TDResNet50((9, 50, 50, 2), components=3, reg=0)
        print(model.summary())
