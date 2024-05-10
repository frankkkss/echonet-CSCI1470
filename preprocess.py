import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import os
import cv2 as cv
import random
import matplotlib.pyplot as plt
import argparse

def preprocess_Stanford_split(des_split='TRAIN', file_path = 'C:/Users/afran/Desktop/BROWN/CSCI_1470_Deep_Learning/Project/Datasets/EchoNet-Dynamic/EchoNet-Dynamic'):
    ## Function to distribute the database as selected by the owners
    # Not used, as we decided to create and randomize our own splits
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


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oscar", action="store_true")
    args = parser.parse_args()
    return args

def fill_array(mask, X1o, X2o, Yo , X1t, X2t, Yt):
    ## Function to fill spaces between rectangles given as labels
    if Yt > Yo:
        # Computing the slopes between vercites of contiguous rectangles to define the limits that need filling
        left_slope = (X1t - X1o)/(Yt - Yo)
        right_slope = (X2t - X2o)/(Yt - Yo)
        for dy in range(Yt - Yo):
            mask[int(np.round(X1o - left_slope*dy, 0)):int(np.round(X2o + right_slope*dy, 0)), Yo + (dy - 1):Yo + dy] = 1
    return mask

def create_labels_for_beat_detection(beats, masks, n_tot):
    ## Creates the binary encoded vectors for labeling the frames of the videos corresponding to EDV or ESV
    vec_systole = vec_dyastole = np.zeros(shape=[n_tot, 128])
    be = np.zeros(shape=[n_tot, 2], dtype=int)                                  # Vector to save tuple of systole and diastole frames, used for testing
    for idx, pat in enumerate(beats):
        # beats[pat]
        m1 = masks[idx, :, :, 0]
        m2 = masks[idx, :, :, 1]
        # The area in the mask was checked as there was no other indication for distinguishing the frames
        # Diastole is the relaxation of the heart, with muscle relaxed, the area of the ventricle is higher
        if np.sum(m1) > np.sum(m2):
            vec_dyastole[idx, beats[pat][0]] = 1
            vec_systole[idx, beats[pat][1]] = 1
            be[idx, :] = np.array([beats[pat][1], beats[pat][0]])
        else:
            vec_dyastole[idx, beats[pat][1]] = 1
            vec_systole[idx, beats[pat][0]] = 1
            be[idx, :] = np.array([beats[pat][0], beats[pat][1]])

    return vec_systole, vec_dyastole, be

def extract_images_for_segment(beats, vids, masks, n_tot):
    ## Constructs the needed segmentation and corresponging images arrays for inputs
    ims_for_masks = np.zeros(shape=[n_tot*2, 112, 112], dtype=np.float16)
    new_masks = np.zeros(shape=[n_tot*2, 112, 112], dtype=np.float16)
    # Shuffle used to improve performance
    # idx_shuf = tf.random.shuffle(range(n_tot))
    for idx, pat in enumerate(beats):
        # Despite shuffle, masks and images are saved in orderly manner by video to keep the systole-diastole relation, so testing is easier later
        new_masks[2*idx, :, :] = masks[idx, :, :, 0]
        new_masks[2*idx + 1, :, :] = masks[idx, :, :, 1]
        ims_for_masks[2*idx, :, :] = vids[idx, :, :, beats[pat][0]]
        ims_for_masks[2*idx + 1, :, :] = vids[idx, :, :, beats[pat][1]]
    return ims_for_masks, new_masks

def main_preprocess(args):
    ## Reading the provided files for the database and extracting the desired data

    # As we made use of the computational power offered by CCv's oscar, the filepaths had to be changed 
    if args.oscar == True:
        file_path = '/oscar/scratch/afranco7/CSCI1470/Datasets/EchoNet-Dynamic/EchoNet-Dynamic'
    else:
        file_path = 'C:/Users/afran/Desktop/BROWN/CSCI_1470_Deep_Learning/Project/Datasets/EchoNet-Dynamic/EchoNet-Dynamic'

    # opening the CSV files containing the information about the videos and the labels
    list_data = pd.read_csv(file_path + '/FileList.csv') 
    vid_path = file_path + '/Videos/'
    seg_file = pd.read_csv(file_path + '/VolumeTracings.csv')

    n = len(list_data['EF'])

    # Cropping videos at 128 frames, so every video which systole or dyastole frames are upwards are discarded
    bin = seg_file['Frame'] < 128
    not_included_masks = {}
    # List of the videos taht are going to be discarded, formed by videos which important frames lie outside of the cutoff 128 frames and those which may appear in only one of the list files
    for b in range(len(seg_file['X1'])):
        if (bin[b] == False) | (seg_file['FileName'][b][:-4] not in np.array(list_data['FileName'])):
            not_included_masks[seg_file['FileName'][b][:-4]] = 0
    n_tot = n - len(not_included_masks.keys()) + 1                                                                                    

    masks = np.zeros([n_tot, 112, 112, 2])
    n_frame_in_vid = None

    prev_img = None
    prev_frame = None
    n_file = 0
    beats = {}

    # Iterating through each of the rows in the second CSV file, containing the segmentations for each selected frame
    for ii in range(len(seg_file['X1'])):
        if seg_file['FileName'][ii][:-4] in not_included_masks.keys():
            pass
        else:
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
                    if (list_data['FrameHeight'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]] != 112) & (list_data['FrameWidth'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]] != 112): 
                        mask_n = cv.resize(mask_n, [112, 112], interpolation=cv.INTER_NEAREST)
                    masks[n_file, :, :, n_frame_in_vid] = mask_n
                    mask_n = np.zeros([list_data['FrameHeight'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]], list_data['FrameWidth'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]]])
                    prev_frame = seg_file['Frame'][ii]
                    beats[seg_file['FileName'][ii][:-4]].append(seg_file['Frame'][ii])
                    n_frame_in_vid = 1
            else:
                if prev_img != None:
                    if (list_data['FrameHeight'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]] != 112) & (list_data['FrameWidth'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]] != 112): 
                        mask_n = cv.resize(mask_n, [112, 112], interpolation=cv.INTER_NEAREST)
                    masks[n_file, :, :, n_frame_in_vid] = mask_n
                    n_file += 1
                mask_n = np.zeros([list_data['FrameHeight'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]], list_data['FrameWidth'][list_data[list_data['FileName'] == seg_file['FileName'][ii][:-4]].index[0]]])
                prev_img = seg_file['FileName'][ii][:-4]
                prev_frame = seg_file['Frame'][ii]
                beats[seg_file['FileName'][ii][:-4]] = [seg_file['Frame'][ii]]
                n_frame_in_vid = 0

    vids_info = []
    vids = np.zeros(shape=[n_tot, 112, 112, 128], dtype=np.uint8)
    n_vid = 0
    EF = np.zeros([n_tot, 1])
    for i in range(n):
        if list_data['FileName'][i] in not_included_masks.keys():
            pass
        else:
            i_vid = cv.VideoCapture(vid_path + list_data['FileName'][i] + '.avi')                                                               # Opening video                                                                                               # Initializing dictionary
            n_frame = 0
            EF[n_vid, 0] = list_data['EF'][i]
            while (i_vid.isOpened()) & (n_frame < 128):                                                                                             
                ret, frame = i_vid.read()                                                                                                       # Getting frames of the vido one by one
                if not ret:                                                                                                                     # ret is true for each frame, when hte video ends and there are no frames it is False
                    # print(f"Stream end. Exiting vid #{i}...")
                    vids_info.append([i, list_data['FileName'][i], 112, 112, list_data['NumberOfFrames'][i]])                                     # Saving information from each of the videos
                    break
                if (list_data['FrameHeight'][i] != 112) & (list_data['FrameWidth'][i] != 112):                                                  # Setting the desired resolution to 112x112
                    frame = cv.resize(frame, (112, 112), interpolation=cv.INTER_CUBIC)                                                            # Resizing the videos that are not in that rsolution
                gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)                                                                                    # Converting to grayscale
                vids[n_vid, :, :, n_frame] = gray
                n_frame += 1
            i_vid.release()
            n_vid += 1
    
    # print(vids.shape)
    
    return vids, beats, masks, EF, n_tot

def splits(args):
    videos, beats, masks, EF, n_tot = main_preprocess(args)
    systole_labels, dyastole_labels, beat_pos = create_labels_for_beat_detection(beats=beats, masks=masks, n_tot=n_tot)
    ims_segm, masks = extract_images_for_segment(beats=beats, vids=videos, masks=masks,n_tot=n_tot)
    videos = np.expand_dims(videos, axis=-1)
    masks = np.expand_dims(masks, axis=-1)
    ims_segm = np.expand_dims(ims_segm, axis=-1)
    # print(f"Shape of vids {videos.shape}\nShape of masks {masks.shape}\nShape of ims {ims_segm.shape}\nShape of labels {systole_labels.shape}")
    
    # idx_shuf = tf.random.shuffle(range(n_tot))
    idx_list = range(n_tot)
    n_train = int(round(n_tot*0.8, 0))
    n_val_test = int(round(n_tot*0.1, 0))
    train_idx = idx_list[:n_train]
    val_idx = idx_list[n_train + 1:n_train + n_val_test]
    test_idx = idx_list[-n_val_test:]

    vids_train = videos[train_idx, :, :, :, :]
    vids_val = videos[val_idx, :, :, :, :]
    vids_test = videos[test_idx, :, :, :, :]
    labels_sys_train = systole_labels[train_idx, :]
    labels_sys_val = systole_labels[val_idx, :]
    labels_sys_test = systole_labels[test_idx, :]
    labels_dyas_train = dyastole_labels[train_idx, :]
    labels_dyas_val = dyastole_labels[val_idx, :]
    labels_dyas_test = dyastole_labels[test_idx, :]
    beat_pos_test = beat_pos[test_idx, :]

    EF_test = EF[test_idx]

    # idx_shuf = tf.random.shuffle(range(n_tot*2))
    n_train = int(round(n_tot*2*0.8, 0))
    n_val_test = int(round(n_tot*2*0.1, 0))
    idx_list = range(n_tot*2)
    train_idx = idx_list[:n_train]
    val_idx = idx_list[n_train + 1:n_train + n_val_test]
    test_idx = idx_list[-n_val_test:]
    
    # EF_shuf = EF[shuf_masks]
    # EF_test = EF_shuf[test_idx]

    mask_train = masks[train_idx, :, :, :]
    mask_val = masks[val_idx, :, :, :]
    mask_test = masks[test_idx, :, :, :]
    ims_train = ims_segm[train_idx, :, :, :]
    ims_val = ims_segm[val_idx, :, :, :]
    ims_test = ims_segm[test_idx, :, :, :]
    return mask_train, mask_val, mask_test, ims_train, ims_val, ims_test, vids_train, vids_val, vids_test, labels_sys_train, labels_sys_val, labels_sys_test, labels_dyas_train, labels_dyas_val, labels_dyas_test, EF_test

if __name__ == "__main__":
    args = parseArguments()
    mtr, mv, mte, imtr, imv, imte, vtr, vv, vte, lstr, lav, lste, ldtr, ldv, ldte, efte = splits(args=args)