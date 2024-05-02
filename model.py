import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Conv3D, Conv2D, Conv2DTranspose, Flatten, MaxPool3D, MaxPool2D, BatchNormalization, ReLU
import numpy as np
tf.compat.v1.enable_eager_execution()


class FrameSelect(Model):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size        # Input size = heigth, width, video_length, channels (since they are grey scale frames the channels will be 1)

        self.conv_block = Sequential([Conv3D(16, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2)), # Si se quiere no reducir el numero de frames agregar: strides= (2, 2, 1), padding= 'SAME'

                                      Conv3D(32, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2)), # Padding valid and stride 1

                                      Conv3D(64, 3, padding= 'SAME'),
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2)),

                                      Conv3D(128, 3, padding= 'SAME'),  # Puede que reduzca las units
                                      BatchNormalization(),
                                      ReLU(),
                                      MaxPool3D((2, 2, 2))
        ])

        self.flatten_layer = Flatten()
        self.dense_block = Sequential([Dense(input_size[2], activation= 'softmax')
                                       ])

    def call(self, videos):
        
        x = self.conv_block(videos)

        x = self.flatten_layer(x)
        probs = self.dense_block(x)

        return probs


class Unet(Model):
    def __init__(self):
        super().__init__()

        self.conv_block1 = Sequential([Conv2D(32, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(32, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU()
         ])
        self.pool_1 = MaxPool2D((2, 2), strides= 2)
        
        self.conv_block2 = Sequential([Conv2D(64, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(64, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU()
         ])
        self.pool_2 = MaxPool2D((2, 2), strides= 2)
        
        self.conv_block3 = Sequential([Conv2D(128, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(128, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU()
         ])
        self.pool_3 = MaxPool2D((2, 2), strides= 2)
        
        self.center = Sequential([Conv2D(256, 3, padding='SAME'),
                                  BatchNormalization(),
                                  ReLU(),
                                  Conv2D(256, 3, padding='SAME'),
                                  BatchNormalization(),
                                  ReLU()
         ])
        
        self.decoder_3 = Conv2DTranspose(128, 2, strides=2)         
        self.decoder_conv_3 = Sequential([Conv2D(128, 3, padding='SAME'),
                                          BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(128, 3, padding='SAME'),
                                          BatchNormalization(),
                                          ReLU()
         ])
        
        self.decoder_2 = Conv2DTranspose(64, 2, strides=2)         
        self.decoder_conv_2 = Sequential([Conv2D(64, 3, padding='SAME'),
                                          BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(64, 3, padding='SAME'),
                                          BatchNormalization(),
                                          ReLU()
         ])
        
        self.decoder_1 = Conv2DTranspose(32, 2, strides=2)         
        self.decoder_conv_1 = Sequential([Conv2D(32, 3, padding='SAME'),
                                          BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(32, 3, padding='SAME'),
                                          BatchNormalization(),
                                          ReLU()
         ])
        
        self.out = Conv2D(1, 1, activation= 'sigmoid')
        
        
    @tf.function
    def call(self, frames):
        
        encoder_1 = self.conv_block1(frames)
        x = self.pool_1(encoder_1)
        encoder_2 = self.conv_block2(x)
        x = self.pool_2(encoder_2)
        encoder_3 = self.conv_block3(x)
        x = self.pool_3(encoder_3)

        x = self.center(x)

        x = self.decoder_3(x)
        x = self.decoder_conv_3(tf.concat([x, encoder_3], axis= -1))
        x = self.decoder_2(x)
        x = self.decoder_conv_2(tf.concat([x, encoder_2], axis= -1))
        x = self.decoder_1(x)
        x = self.decoder_conv_1(tf.concat([x, encoder_1], axis= -1))
        
        out = self.out(x)

        return out