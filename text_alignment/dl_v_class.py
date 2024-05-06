from torch.utils.data import Dataset, Subset
import numpy as np
import torch
from PIL import Image
import zipfile
import tempfile
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
from transformers import AutoProcessor
import os
from scipy.interpolate import interp1d
import numpy as np
import torch
from scipy.interpolate import interp1d
from collections import defaultdict


class EEGDataProcessor:
    def __init__(self, eeg_signals_path,participant_number=4):
        # Initialize the processor with file paths and parameters.
        data_dict = torch.load(eeg_signals_path)
        data = [data_dict['dataset'][i] for i in range(len(data_dict['dataset']) ) if data_dict['dataset'][i]['subject']==participant_number]

        self.labels = data_dict["labels"]
        self.images = data_dict["images"]
        #self.image_path = 'data/stimuli.zip'

        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
        self.tmp_path = None
        self.zip_extracted = False

        self.data_len = 512

        self.data = data


    def __getitem__(self,i):
        # Load and standardize raw EEG data.

        eeg = self.data[i]["eeg"].float().t()

        # Some of the data is less than 512ms for an epoch so we need to interpolate
        #eeg =eeg.permute(1,0)
        eeg = eeg[20:480,:]
        eeg = np.array(eeg.transpose(0,1))
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, self.data_len)
        f = interp1d(x, eeg)
        eeg = f(x2)
        eeg = torch.from_numpy(eeg).float()

      
        
        caption = self.data[i]["caption"]
        
        if caption is None:
            return None
        


        return {"eeg": eeg/10, 'caption': caption} 
        

    def __len__(self):
        return len(self.data)


def load_and_preprocess_eeg_data(eeg_signals_path, participant_number):
    dataset = EEGDataProcessor(eeg_signals_path, participant_number)
    return dataset

if __name__ == "__main__":
    
    dataset = load_and_preprocess_eeg_data("./data/eeg2.pth", 4)

    print(len(dataset))
    
    # Example usage of the dataset
    sample = dataset[431]
    print(sample['eeg'].shape)
    print(sample['caption'])

    


    #plt.plot(sample['eeg'][121].numpy()) # Plot a single EEG channel
    #plt.show()