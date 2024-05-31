import os
import sys
#set the path to the EEG folder
#sys.path.append('/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/JEPA/dataloader')
import subprocess
import time
from src.datasets import dataset as ds
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from logging import getLogger

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()
from torch.utils.data import DataLoader


#we weill need to return 

# 1. A dataset object
# 2. A dataloader object
# 3. A distributed sampler object

def load_eeg_data(paricipant_subset,
                  batch_size=1,
                  collator=None,
                  num_workers=4,
                  pin_mem=True):
    paths_EEG = []
    for each in paricipant_subset:
        
        root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
        #root = "/home/hutchins/projects/def-yalda/hutchins/data/sub-"
        mid = "/eeg/sub-"
        end = "_task-rsvp_eeg.vhdr"
        
        if each < 10:
            path = root + "0" + str(each ) + mid + "0" + str(each ) + end
        else:
            path = root + str(each ) + mid + str(each ) + end

        paths_EEG.append(path)
    
    datasets = []
    for path in paths_EEG:
        dataset = ds.create_dataset(raw_file_path=path, 
                                         event_description='Event/E  1', 
                                         batch_size=batch_size)
        datasets.append(dataset)

    # Combine all individual datasets into a single ConcatDataset
    combined_dataset = ConcatDataset(datasets)
    
    #split this into a train and validation dataset
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])
    
    

    
    
    # Use a simple DataLoader without a sampler for non-distributed training
    dataloader_eeg_train = DataLoader(train_dataset,
                                collate_fn=collator,
                                batch_size=batch_size,
                                shuffle=True,  # Assuming you want to shuffle for training
                                num_workers=num_workers,
                                pin_memory=pin_mem)
    dataloader_eeg_val = DataLoader(val_dataset,
                                collate_fn=collator,
                                batch_size=batch_size,
                                shuffle=False,  # No need to shuffle for validation
                                num_workers=num_workers,
                                pin_memory=pin_mem)
    
    return combined_dataset, dataloader_eeg_train, dataloader_eeg_val