import tensorflow as tf
from keras import Sequential, Model
from keras.layers import Dense, Conv3D, Conv2D, Conv1D, Conv2DTranspose, Dropout, Flatten, MaxPool3D, MaxPool2D, BatchNormalization, ReLU
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
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        self.conv_block1 = Sequential([Conv2D(32, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(32, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       MaxPool2D((2, 2))
         ])
        
        self.conv_block2 = Sequential([Conv2D(64, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(64, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       MaxPool2D((2, 2))
         ])
        
        self.conv_block3 = Sequential([Conv2D(128, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(128, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       MaxPool2D((2, 2))
         ])
        
        self.center = Sequential([Conv2D(256, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU(),
                                       Conv2D(256, 3, padding='SAME'),
                                       BatchNormalization(),
                                       ReLU()
         ])
        
        self.decoder_3 = Conv2DTranspose(128, 2, strides=2, padding='SAME')         
        self.decoder_conv_3 = Sequential([BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(128, 3, padding='SAME'), 
                                          BatchNormalization(),
                                          ReLU(),
                                          Conv2D(128, 3, padding='SAME')
         ])
        
        self.decoder_2 = Conv2DTranspose(64, 2, strides=2, padding='SAME')         
        self.decoder_conv_2 = Sequential([BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(64, 3, padding='SAME'), 
                                          BatchNormalization(),
                                          ReLU(),
                                          Conv2D(128, 3, padding='SAME')
         ])
        
        self.decoder_1 = Conv2DTranspose(32, 2, strides=2, padding='SAME')         
        self.decoder_conv_1 = Sequential([BatchNormalization(),     # recives a concatenated input 
                                          ReLU(),
                                          Conv2D(32, 3, padding='SAME'), 
                                          BatchNormalization(),
                                          ReLU(),
                                          Conv2D(32, 3, padding='SAME')
         ])
        
    @tf.function
    def call(self, frames):
        
        x, encoder_1 = self.conv_block1(frames)
        x, encoder_2 = self.conv_block2(x)
        x, encoder_3 = self.conv_block3(x)

        x = self.center(x)

        x = self.decoder_3(x)
        x = self.decoder_conv_3(tf.concat([x, encoder_3]))
        x = self.decoder_2(x)
        x = self.decoder_conv_2(tf.concat([x, encoder_2]))
        x = self.decoder_1(x)
        x = self.decoder_conv_1(tf.concat([x, encoder_1]))

        return x


"""
def conv_block(input_tensor, num_filters):
    encoder = layers.Conv3D(num_filters, (3, 3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.PReLU()(encoder)
    encoder = layers.Dropout(0.3)(encoder)  # Añade dropout
    return layers.PReLU()(encoder)

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    return encoder, layers.MaxPooling3D((2, 2, 2))(encoder)

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.PReLU()(decoder)

    return layers.Conv3D(num_filters, (3, 3, 3), padding='same')(decoder)


def get_vnet(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    encoder0, pool0 = encoder_block(inputs, 32)
    
    center = conv_block(pool0, 64)
    
    decoder0 = decoder_block(center, encoder0, 32)

    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation='softmax')(decoder0)"""