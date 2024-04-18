import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import os
import cv2 as cv
import random
import matplotlib.pyplot as plt

def preprocess(des_split='TRAIN', file_path = 'C:/Users/afran/Desktop/BROWN/CSCI_1470_Deep_Learning/Project/Datasets/EchoNet-Dynamic/EchoNet-Dynamic'):

    des_split.upper()
    
    vid_path = file_path + '/Videos/'
    list_data = pd.read_csv(file_path + '/FileList.csv')

    split_filename = []
    split_file = {}
    
    for n, split in enumerate(list_data['Split']):
        if split == des_split:
            split_filename.append(list_data['FileName'][n])
            split_file[list_data['FileName'][n]] = cv.VideoCapture(vid_path + split_filename[-1] + '.avi')
            # cv.cvtColor(, 6)


    seg_data = pd.read_csv(file_path + '/VolumeTracings.csv')

    # for n, split in enumerate(list_data['Split']):

    
    # pil.Image.open().convert('L')

# preprocess(des_split='VAL')

def fill_array(mask, X1o, X2o, Yo , X1t, X2t, Yt):
    if Yt > Yo:
        Yt = int(np.round(Yt, 0))
        Yo = int(np.round(Yo, 0))
        left_slope = (X1t - X1o)/(Yt - Yo)
        right_slope = (X2t - X2o)/(Yt - Yo)
        for dy in range(Yt - Yo):
            mask[int(np.round(X1o - left_slope*dy, 0)):int(np.round(X2o + right_slope*dy, 0)), Yo:Yo + dy] = 1
    return mask


file_path = 'C:/Users/afran/Desktop/BROWN/CSCI_1470_Deep_Learning/Project/Datasets/EchoNet-Dynamic/EchoNet-Dynamic'                     # Path to your dataset
list_data = pd.read_csv(file_path + '/FileList.csv')                                                                                    # 
vid_path = file_path + '/Videos/'

vids = {}
vids_info = []

n = len(list_data['EF'])
# i = 0
for i in range(n):
    i_vid = cv.VideoCapture(vid_path + list_data['FileName'][i] + '.avi')                                                               # Opening video
    vids[list_data['FileName'][i]] = []                                                                                                 # Initializing dictionary
    while i_vid.isOpened():                                                                                             
        ret, frame = i_vid.read()                                                                                                       # Getting frames of the vido one by one
        if not ret:                                                                                                                     # ret is true for each frame, when hte video ends and there are no frames it is False
            print("Stream end. Exiting ...")
            vids_info.append([list_data['FileName'][i], 112, 112, len(vids[list_data['FileName'][i]])])                                 # SSaving information from each of the videos
            break
        if (list_data['FrameHeight'][i] != 112) & (list_data['FrameWidth'][i] != 112):                                                  # Setting the desired resolution to 112x112
            gray = cv.resize(gray, [112, 112], interpolation=INTER_CUBIC)                                                               # Resizing the videos that are not in that rsolution
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)                                                                                    # Converting to grayscale
        vids[list_data['FileName'][i]].append(gray)                                                                                     # Adding each frame to the dictionary
        # vids_info.append([list_data['FileName'][i], 112, 112, len(vids[list_data['FileName'][i]])])                                   # 
        # cv.imshow('frame', gray)
        # cv.waitKey(25)
    i_vid.release()
    # cv.destroyAllWindows()

seg_file = pd.read_csv(file_path + '/VolumeTracings.csv')                                                                               # Opening the labels file
seg_data = {}

prev_img = None
prev_frame = None
# 41
n_file = 0
for ii in range(len(seg_file['X1'])):
    if prev_img != None:
        prev_Mx = Mx
        prev_mx = mx
        prev_y = My
    Mx = np.max(np.array([int(np.round(seg_file['X1'][ii],0)), int(np.round(seg_file['X2'][ii],0))]))
    mx = np.min(np.array([int(np.round(seg_file['X1'][ii],0)), int(np.round(seg_file['X2'][ii],0))]))
    My = np.max(np.array([int(np.round(seg_file['Y1'][ii],0)), int(np.round(seg_file['Y2'][ii],0))]))
    my = np.min(np.array([int(np.round(seg_file['Y1'][ii],0)), int(np.round(seg_file['Y2'][ii],0))]))
    if seg_file['FileName'][ii][:-4] == prev_img:
        if seg_file['Frame'][ii] == prev_frame:
            mask_n[my:My, mx:Mx] = 1
            if prev_y <= my:
                mask_n = fill_array(mask_n, prev_mx, prev_Mx, prev_y, mx, Mx, my)
        else:
            seg_data[prev_img].append([prev_frame, mask_n])
            mask_n = np.zeros([list_data['FrameHeight'][n_file], list_data['FrameWidth'][n_file]])
            prev_frame = seg_file['Frame'][ii]
            mask_n[my:My, mx:Mx] = 1
    else:
        if prev_img != None:
            seg_data[prev_img].append([prev_frame, mask_n])
            n_file += 1
        mask_n = np.zeros([list_data['FrameHeight'][n_file], list_data['FrameWidth'][n_file]])
        prev_img = seg_file['FileName'][ii][:-4]
        prev_frame = seg_file['Frame'][ii]
        seg_data[prev_img] = []
        mask_n[my:My, mx:Mx] = 1
        
# Optimal clip length: 70 frames
            
        


#i_vid = cv.cvtColor(i_vid, cv.COLOR_RGB2GRAY)
#i_vid = cv.resize(i_vid, [112, 112, :], interpolation=INTER_CUBIC)
#vids.append(i_vid)




#idx_shuf = random.shuffle(range(n))
#n_train = n*0.75
#n_val_test = n*0.125
#train_idx = idx_shuf[:n_train]
#val_idx = idx_shuf[n_train + 1:n_train + n_val_test]
#test_idx = idx_shuf[-n_val_test:]
