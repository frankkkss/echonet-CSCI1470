import numpy as np
import pandas as pd
#import tensorflow as tf
import os

file_path = r'C:\Users\afran\Desktop\BROWN\CSCI 1470 Deep Learning\Project\Datasets\EchoNet-Dynamic\EchoNet-Dynamic'
vid_path = os.path.join(file_path, r'\Videos')

file_data = pd.read_csv(file_path + r'\FileList.csv')

seg_path = os.path.join(file_path, r'\VolumeTracing.csv')

