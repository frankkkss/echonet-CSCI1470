import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Conv3D, Dropout, Flatten, MaxPool3D, BatchNormalization, ReLU
import numpy as np


class FrameSelect(Model):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size        # Input size = hiegth, width, video_length, channels (since they are grey scale frames the channels will be 1)

        self.conv_block = Sequential([Conv3D(16, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2), strides= (2, 2, 2), padding= 'SAME'), 

                                      Conv3D(32, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2), strides= (2, 2, 2), padding= 'SAME'),

                                      Conv3D(64, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2), strides= (2, 2, 2), padding= 'SAME'),

                                      Conv3D(128, 3, padding= 'SAME'),       # Puede que reduzca las units
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2), strides= (2, 2, 2), padding= 'SAME')
        ])

        self.flatten_layer = Flatten()
        self.dense_block = Sequential([Dense(256, activation= 'leaky_relu'),
                                       Dense(input_size[2], activation= 'softmax')
                                       ])

    def call(self, videos):
        
        x = self.conv_block(videos)

        x = self.flatten_layer(x)
        probs = self.dense_block(x)

        return probs


class Unet(Model):
    pass