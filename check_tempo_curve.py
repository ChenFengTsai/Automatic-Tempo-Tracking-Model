# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:16:41 2021

@author: Sunny
"""

#%%
import os
import numpy as np

import librosa
from utils_comparative_0604 import * 
import glob
import matplotlib.pyplot as plt

beat_dir = '/home/rich.tsai/NAS_189/home/BeatTracking/Past/datasets/ASAP/asap-dataset-1.1/downbeats/'
annotation_files = glob.glob(os.path.join(beat_dir, "*.beats"))

check_num = 5
for ann_file in annotation_files:
    # break
    check_num -=1
    if check_num ==0:
        break
    beats_ann = np.loadtxt(ann_file)
    groundtruth_tempo_curve = tempoWin(beats_ann, win_len = 12, hop = 1)
    
    songname = os.path.basename(ann_file).replace('.beats', '')
    
    ## plot to see
    plt.figure()
    plt.plot(groundtruth_tempo_curve, color = 'red', label = 'groundtruth tempo')
    plt.ylim([10, 300])
    plt.xlabel('time (sec)')
    plt.ylabel('tempo (bpm)')
    plt.title(songname)