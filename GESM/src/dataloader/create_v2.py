from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from logging import getLogger
import torch
import mne
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
participant_list = [2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13, 14, 15, 16, 17, 19, 20,21, 22, 24, 25, 26, 27, 28, 29, 30,31, 32, 33, 34, 35, 36, 37, 38, 39, 40,41, 42, 43, 44, 45, 46, 47, 48]

def load_eeg_data(paricipant_subset,
                  ):
    paths_EEG = []
    paths_events = []
    for each in paricipant_subset:
        
        root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
        #root = "/home/hutchins/projects/def-yalda/hutchins/data/sub-"
        mid = "/eeg/sub-"
        end = "_task-rsvp_eeg.vhdr"
        
        if each < 10:
            path = root + "0" + str(each ) + mid + "0" + str(each ) + end
            path_event = root + "0" + str(each ) + mid + "0" + str(each ) + "_task-rsvp_events.csv"
        else:
            path = root + str(each ) + mid + str(each ) + end
            path_event = root + str(each ) + mid + str(each ) + "_task-rsvp_events.csv"
        
        paths_events.append(path_event)

        paths_EEG.append(path)
    
    return paths_EEG, paths_events

print(len(participant_list))
paths_EEG, paths_events = load_eeg_data(participant_list)


input_folders = paths_EEG
output_folder = '/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/GESM/src/dataloader/gen_data'


def load_raw_data(raw_file_path):
        raw = mne.io.read_raw_brainvision(raw_file_path, preload=True)

        raw.crop(tmax=2000, tmin=1950)
        
        return raw
        
def preprocess_raw_data(raw_data):
        raw_data.set_eeg_reference('average', projection=True)  
        raw_data.filter(5., 95., fir_design='firwin')
        
        return raw_data 
        

def extract_epochs(raw, event_descriptions, tmin=-0.128, tmax=0.512):
        event_description = event_descriptions
        print("extracting epochs...")
        raw._data = raw._data.astype(np.float32)
        events, event_dict = mne.events_from_annotations(raw, event_id=None, regexp=event_description)
        print(f"size of events: {events.shape}")
        print(f"event dict: {event_dict}")
        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)
        kept_events = epochs.events[:, 2]
        kept_events_ids_set = set(kept_events)
        
        return epochs, events, kept_events
    
    
    

print(len(paths_EEG))
test_path = paths_EEG[12]
event_path = paths_events[12]




event_df = pd.read_csv(event_path)

# remove all rows where object number is -1 
event_df = event_df[event_df['objectnumber'] != -1]

my_annotations = mne.Annotations(onset=event_df['time_stimon'], 
                                 duration=event_df['stimdur'], 
                                 description=event_df['objectnumber'])

print(my_annotations)

descriptions = event_df['objectnumber']


raw_data = load_raw_data(test_path)
raw_data.set_annotations(my_annotations)

#extract epochs using these annotations
epochs = extract_epochs(raw_data, descriptions, tmin=-0.128, tmax=0.512,)

print(len(epochs))

