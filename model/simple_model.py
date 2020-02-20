# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:48 2020

定义一个简单的VGG神经网络，方便之后调用

@author: ndq
"""

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K


class SimpleModel:
    def build(height, width, channels, classes):
        '''1.初始化一个序列模型，根据使用后端的不同（TensorFlow或者Theano），
        inputshape分为h*w*c或者c*h*w；
        2.在模型中添加层，隐藏层主要包括卷积层，池化层，卷积层一般使用3*3的卷积核，
        激活函数基本为relu，在使用完激活函数后一般会进行BN标准化操作，
        池化层一般size为2*2;
        3.卷积层的权重在全连接层前，需要进行Flatten操作;
        4.输出层为全连接层，多分类的激活函数为softmax
        '''
        model = Sequential()
        inputShape = (height, width, channels)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (channels, height, width)
            chanDim = 1        
        
        # 添加层，按照CONV => RELU => POOL设置，并设置dropout
        model.add(Conv2D(32, (3, 3), padding='same', 
            input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 添加层，按照(CONV => RELU) * 2 => POOL设置
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 添加层，按照(CONV => RELU) * 3 => POOL设置
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # 添加层，设置全连接层FC => RELU
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # 添加输出层，使用softmax作为激活函数
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        
        return model
        
        
        
        
        
        
        
        
        
        
        
        
        
        