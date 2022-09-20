import os
import numpy as np
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate,Conv2DTranspose,add,Concatenate,Add,Subtract
from tensorflow.keras.layers import Input,Conv2D,concatenate,MaxPooling2D,BatchNormalization
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def standard_unit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x0 = x
    # x = Dropout(0.2, name='dp' + stage + '_1')(x)
    x = BatchNormalization(name='bn' + stage + '_1')(x)  # much better than dropout
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='selu', name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    # x = Dropout(0.2, name='dp' + stage + '_2')(x)
    x = BatchNormalization(name='bn' + stage + '_2')(x)
    if mode == 'residual':
        # x=Add(name='resi'+stage)([x,input_tensor])# 维度不相同！
        x = Add(name='resi' + stage)([x, x0])

    return x

def Nest_Net2(input_shape, num_class=1, deep_supervision=False):
    nb_filter = [32, 64, 128, 256, 512]
    # nb_filter = [16, 32, 64, 128, 256]
    mode = 'residual'  # mode='residual' seems to improve better than DS
    # Handle Dimension Ordering for different backends
    bn_axis = 3
    inputs = Input(shape=input_shape)

    conv1_1 = standard_unit(inputs, stage='11', nb_filter=nb_filter[0])  # add 要求输入输出维度相同
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)  # (?,128,128,32)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)  # (?,64,64,64)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)  # (?,256,256,64)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0], mode=mode)  # (?,256,256,32)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1], mode=mode)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0], mode=mode)  # (?,256,256,32)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 =standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2], mode=mode)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1], mode=mode)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 =standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0], mode=mode)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4], mode=mode)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3], mode=mode)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2], mode=mode)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1], mode=mode)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 =standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0], mode=mode)

    nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_1',
                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_2',
                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_3',
                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_4',
                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)
    # using combined loss
    conv_fuse = concatenate([conv1_2, conv1_3, conv1_4, conv1_5], name='merge_fuse', axis=bn_axis)

    nestnet_output_5 = Conv2D(num_class, (1, 1), activation='sigmoid', name='output_5',
                              kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv_fuse)

    if deep_supervision:
        model = tf.keras.Model(input=inputs, output=[nestnet_output_1,
                                            nestnet_output_2,
                                            nestnet_output_3,
                                            nestnet_output_4, nestnet_output_5])

    else:
        model = tf.keras.Model(inputs = inputs, outputs = nestnet_output_5)
    return model

# if __name__ == '__main__':
#     assert True == tf.executing_eagerly();
#     input_shape = [256, 256, 6]
#     model=Nest_Net2(input_shape,2)
#     results = model(tf.constant(np.random.normal(size = (1, 256, 256, 6)), dtype = tf.float32));
#     print(results)
