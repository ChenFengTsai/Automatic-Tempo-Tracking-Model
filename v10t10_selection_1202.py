# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:54:59 2020

@author: CITI
"""

#%%
import os
import pandas as pd
import numpy as np

dataset_dir = '/home/sunnycyc/NAS_189/home/BeatTracking/BeatData2/asap-dataset-master/'
meta_path = os.path.join(dataset_dir, "metadata.csv")

df = pd.read_csv(meta_path)
df_withaudio = df.loc[df["audio_performance"].notnull()]
unique_df = df_withaudio.drop_duplicates(subset = ["title", "composer"])
#unique_df = df.drop_duplicates(subset = ["title", "composer"])
#unique_df_withaudio = unique_df.loc[unique_df["audio_performance"].notnull()]

#####+++ calculate performances number of each unique piece of work
#unique_dict_list = unique_df_withaudio.to_dict('records')
unique_dict_list = unique_df.to_dict('records')

org_list = []
song_num = 0
for song_dict in unique_dict_list:
#    break
    audio_performances = df_withaudio.loc[(df_withaudio["title"]==song_dict["title"]) & \
                                (df_withaudio["composer"]==song_dict["composer"])]
    song_num +=len(audio_performances)
    org_list.append([ len(audio_performances),song_dict, audio_performances])
#%%
####+++ perform random selection

import random

def randomSelect(org_list, num_to_select = 10):
    s_list = [] # selected unique song pieces
    s_songnum = 0 # selected song num
    while s_songnum < num_to_select:
        rand_ind = random.randint(0, len(org_list)-1)
        performance_num = org_list[rand_ind][0] # num of performances for that song
        if performance_num <= num_to_select-s_songnum:
            s_list.append(org_list.pop(rand_ind))
            s_songnum += performance_num
        else:
            pass
    return s_list

valid_list = randomSelect(org_list, num_to_select = 20)
test_list = randomSelect(valid_list, num_to_select = 10)
    

#%%
####+++ save v10t10_valid, train audio files.txt
asap_main_dir = '/home/sunnycyc/NAS_189/home/BeatTracking/BeatData2/asap-dataset-master/'
def extract_audiofiles(valid_list):
    audio_paths = []
    for song_num, song_dict, performances in valid_list:
#        break
        audio_performances = performances["audio_performance"].tolist()
        audio_paths += [os.path.join(asap_main_dir, i) for i in audio_performances]
    return audio_paths

def existenceCheck(audio_paths):
    for audiopath in audio_paths:
        if not os.path.exists(audiopath):
            print("can't find:", audiopath)

train_list = info['train_list']
valid_paths = extract_audiofiles(valid_list)
test_paths = extract_audiofiles(test_list)
train_paths = extract_audiofiles(train_list)
existenceCheck(valid_paths+test_paths+train_paths)
#%%
####+++ save here

def savetxt(audiopaths, sname = "v10t10_valid.txt", main_dir = asap_main_dir):
    g = open(os.path.join(main_dir, sname), 'w')
    for eachsong in audiopaths:
        g.write(eachsong+'\n')
    g.close()
    
    
savetxt(train_paths, sname = "v10t10_train_audiofiles.txt")
savetxt(valid_paths, sname = "v10t10_valid_audiofiles.txt")
savetxt(test_paths, sname = "v10t10_test_audiofiles.txt")


#%%
####+++ save curretn split for future checking
from pathlib import Path
info_save_dir = os.path.join('/home/sunnycyc/NAS_189/home/BeatTracking/BeatData2/', 
                             "ASAP_split_info")
if not os.path.exists(info_save_dir):
    Path(info_save_dir).mkdir(parents = True, exist_ok = True)
    
info_dict = {
        'train_list': org_list,
        'valid_list': valid_list, 
        'test_list': test_list}
import pickle
pickle_sname = os.path.join(info_save_dir, "v10t10_info_1202.pickle")
with open(pickle_sname, 'wb') as file:
    pickle.dump(info_dict, file)

#%%
with open(pickle_sname, 'rb') as file:
    info = pickle.load(file)

