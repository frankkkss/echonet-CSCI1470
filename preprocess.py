import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import os
import cv2

def preprocess(des_split='TRAIN', file_path = r'C:\Users\afran\Desktop\BROWN\CSCI 1470 Deep Learning\Project\Datasets\EchoNet-Dynamic\EchoNet-Dynamic'):

    des_split.upper()
    
    vid_path = file_path + r'\Videos\'
    file_data = pd.read_csv(file_path + r'\FileList.csv')

    split_filename = []
    split_file = {}
    
    for n, split in enumerate(file_data['Split']):
        if split == des_split:
            split_filename.append(file_data['FileName'][n])
            split_file[file_data['FileName'][n]] = cv2.VideoCapture(vid_path + split_filename[-1] + '.avi')
            cv2.cvtColor(, 6)


    seg_data = pd.read_csv(file_path + r'\VolumeTracings.csv')

    # for n, split in enumerate(file_data['Split']):

    
    # pil.Image.open().convert('L')

preprocess(des_split='VAL')