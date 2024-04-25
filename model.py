import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPool3D, BatchNormalization, ReLU
import numpy as np


class FrameSelect(Model):
    def __init__(self, input_size):
        self.input_size = input_size        # Input size = hiegth, width, video_length, channels (since they are grey scale frames the channels will be 1)

        self.conv_1 = Conv3D(32, 5, padding= 'SAME')
        self.conv_2 = Conv3D(64, 5, padding= 'SAME')
        self.conv_3 = Conv3D(128, 3, padding= 'SAME')
        self.conv_4 = Conv3D(256, 3, padding= 'SAME')

        self.norm_pool = Sequential([BatchNormalization(),
                                     ReLU(),
                                     MaxPool3D((2, 2, 2), strides= (2, 2, 1), padding= 'SAME')])
                                      
        self.flatten_layer = Flatten()
        self.dense_block = Sequential([Dense(128, activation= 'leaky_relu'),
                                       Dropout(0.3),
                                       Dense(input_size[3], activation= 'softmax')
                                       ])

    def call(self, videos):
        
        x1 = self.conv_1(videos)
        x1 = tf.add(x1, videos)
        x1 = self.norm_pool(x1)

        x2 = self.conv_1(x1)
        x2 = tf.add(x2, x1)
        x2 = self.norm_pool(x2)

        x3 = self.conv_1(x2)
        x3 = tf.add(x3, x2)
        x3 = self.norm_pool(x3)

        x4 = self.conv_1(x3)
        x4 = tf.add(x4, x3)
        x4 = self.norm_pool(x4)

        x = self.flatten_layer(x4)
        probs = self.dense_block(x)

        return probs


class Unet(Model):
    pass