import mne
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from PIL import Image
import requests
import re 
from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor
from accelerate import Accelerator

"""Currenlty cropping the dataset for testing in line 74"""

# Set seed
torch.manual_seed(42)

class EEGDataProcessor:
    def __init__(self, raw_file_path, event_description, resample_factor=3):
        self.raw_file_path = raw_file_path  
        self.event_description = event_description 
        self.resample_factor = resample_factor
        existing_channel_locations = pd.read_csv("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model/electrode_locations/old_points.csv")
        new_channel_locations = pd.read_csv("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model/electrode_locations/new_points.csv")
        
        #covert both to cartesian
        new_points_cartesian = np.array([self.spherical_to_cartesian(row['radius'], row['theta'], row['phi']) for index, row in new_channel_locations.iterrows()])
        old_points_cartesian = np.array([self.spherical_to_cartesian(row['radius'], row['theta'], row['phi']) for index, row in existing_channel_locations.iterrows()])

        interpolation_weights = []

        # Iterate over each new point to find the closest old points and calculate weights
        for new_point in new_points_cartesian:
            closest_indices, closest_distances = self.find_closest_points(new_point, old_points_cartesian)
            weights = self.calculate_weights(closest_distances)
            interpolation_weights.append((closest_indices, weights))
            
        self.interpolation_weights = interpolation_weights
        self.new_points_cartesian = new_points_cartesian
        self.old_points_cartesian = old_points_cartesian
        self.accelerator = Accelerator()
        #self.image_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("text encoder created")
        self.tokenized_descriptions = None
        self.text_encoder.to(self.accelerator.device)
        #self.processor.to(self.accelerator.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)
        
        
        file_name = raw_file_path.split('/')[-1]
        
        pattern = r'sub-\d+'
        
        match = re.search(pattern, file_name)
        
        if match:
            extracted_part = match.group()
            print("Extracted part:", extracted_part)
        else:
            print("No match found.")
        
        self.csv_file_name = "./csv_files/" + extracted_part + "_task-rsvp_events.csv"
        
        # check that that is infact a path
    
        
        
        
        
        #/mnt/a/MainFolder/Neural Nirvana/Data/sub-02/eeg/sub-02_task-rsvp_eeg.vhdr"
        
    
                
        
    def find_closest_points(self, new_point, old_points):
        """
        Find the closest two old points to the new point.
        Returns the indices of the two closest points and their distances.
        """
        distances = np.linalg.norm(old_points - new_point, axis=1)
        closest_indices = np.argsort(distances)[:2]
        return closest_indices, distances[closest_indices]
    
    def calculate_weights(self, distances):
        """
        Calculate weights for interpolation based on distances.
        Weights are inversely proportional to the distance.
        """
        weights = 1 / distances
        normalized_weights = weights / np.sum(weights)
        return normalized_weights
    
    def spherical_to_cartesian(self, r, theta, phi):
        """
        Convert spherical coordinates to Cartesian coordinates.
        theta and phi should be in degrees.
        """
        theta_rad = np.deg2rad(theta)
        phi_rad = np.deg2rad(phi)
        x = r * np.sin(theta_rad) * np.cos(phi_rad)
        y = r * np.sin(theta_rad) * np.sin(phi_rad)
        z = r * np.cos(theta_rad)
        return x, y, z
    
    
    def load_raw_data(self):
        self.raw = mne.io.read_raw_brainvision(self.raw_file_path, preload=True)
        self.current_sfreq = self.raw.info["sfreq"]

        self.raw.crop(tmax=2000, tmin=1950)
        
        original_data = self.raw.get_data()
        
        num_original_channels = original_data.shape[0]
        num_new_channels = 128 - num_original_channels  
        
        interpolated_data = np.zeros((num_new_channels, original_data.shape[1]))
        for index, (indices, weights) in enumerate(self.interpolation_weights[:num_new_channels]):
            for i, weight in zip(indices, weights):
                interpolated_data[index] += original_data[i] * weight
        
        concatenated_data = np.vstack((original_data, interpolated_data))
        self.raw._data = concatenated_data        
        
        concatenated_data = concatenated_data * 10000
       
        num_new_channels = interpolated_data.shape[0]  
        new_ch_names = ['IntCh' + str(i) for i in range(num_new_channels)]
        new_ch_types = ['eeg'] * num_new_channels 
        new_ch_info = mne.create_info(ch_names=new_ch_names, sfreq=self.raw.info['sfreq'], ch_types=new_ch_types)
    
        interpolated_raw = mne.io.RawArray(interpolated_data, new_ch_info)
        
        self.raw.add_channels([interpolated_raw], force_update_info=True)

        self.raw._data = concatenated_data
        
        
    def preprocess_raw_data(self):
        self.raw.set_eeg_reference('average', projection=True)  
        self.raw.filter(5., 95., fir_design='firwin') 

    def extract_epochs(self, tmin=-0.128, tmax=0.512):
        print("Extracting epochs...")

        # Load descriptions from CSV
        csv_file_path = self.csv_file_name
        self.image_descriptions = pd.read_csv(csv_file_path)
        
        # Ensure data is in the correct format
        self.raw._data = self.raw._data.astype(np.float32)

        # Extract events and create epochs
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=self.event_description)
        
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)
        print(f"epoch events: {events.shape}")
        # Identify kept events
        kept_events = self.epochs.events[:, 2]
        kept_event_ids_set = set(kept_events)
        self.image_descriptions['eventnumber'] = self.image_descriptions['eventnumber'].astype(int)

        # Filter descriptions based on kept events
        filtered_df = self.image_descriptions[self.image_descriptions['eventnumber'].isin(kept_event_ids_set)]
        self.descriptions = filtered_df['caption'].tolist()
        
        print(f"Descriptions length: {len(self.descriptions)}")
        
        # Tokenize descriptions
        tokenized_inputs = [self.processor(desc, return_tensors="pt", padding=True, truncation=True) for desc in self.descriptions]

        # Prepare and encode descriptions
        self.tokenized_descriptions = []
        for inputs in tokenized_inputs:
            # Move inputs to the correct device
            inputs = {key: value.to(self.accelerator.device) for key, value in inputs.items()}
            
            # Encode using the text encoder
            encoded_description = self.text_encoder(**inputs)
            self.tokenized_descriptions.append(encoded_description)

        print("Epochs and descriptions extracted and tokenized.")
            
        

        
        

    def process_eeg_data(self):
        self.load_raw_data()
        self.preprocess_raw_data()
        self.extract_epochs()


class EEGDataLoader:
    def __init__(self, epochs, tokenized_descriptions):
        eeg_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        print(eeg_data.shape)
        eeg_data = eeg_data[:, :, :512]
        self.n_channels, self.n_times = eeg_data.shape[1], eeg_data.shape[2]
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        
        # Assuming we use the last_hidden_state for the description features
        # Check and use pooler_output if available, else use last_hidden_state
        features = []
        for td in tokenized_descriptions:
            if hasattr(td, 'pooler_output') and td.pooler_output is not None:
                features.append(td.pooler_output)
            else:
                # Taking the mean of the last_hidden_state as a simple way to get a fixed-size vector
                # You can choose a different method based on your needs
                features.append(td.last_hidden_state.mean(dim=1))
        text_features_tensor = torch.stack(features)
        
        #print the sizes of the tensors
        print(f"EEG Data Tensor Size: {eeg_data_tensor.shape}")
        print(f"Text Features Tensor Size: {text_features_tensor.shape}")
        
        #print the shapes of the tensors
        print(f"EEG Data Tensor Shape: {eeg_data_tensor.shape}")
        print(f"Text Features Tensor Shape: {text_features_tensor.shape}")

        self.dataset = TensorDataset(eeg_data_tensor, text_features_tensor)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset[index]
        print(f"Item type: {type(item)}, Item content: {item}")
        eeg, text_feature = item  # If this line throws, the print above tells us what's wrong
        return {'eeg': eeg, 'text_feature': text_feature}
        
def create_dataset(raw_file_path, event_description, batch_size):
    # Assuming EEGDataProcessor is defined elsewhere and works as intended
    eeg_processor = EEGDataProcessor(raw_file_path, event_description)
    eeg_processor.process_eeg_data()
    dataset = EEGDataLoader(eeg_processor.epochs, eeg_processor.tokenized_descriptions)
    
    return dataset

def shuffle_dataset(dataset):
    # Manually shuffle the dataset
    indices = torch.randperm(len(dataset)).tolist()
    dataset.dataset = dataset.dataset[indices]
    return dataset

if __name__ == "__main__":
    raw_file_path = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-02/eeg/sub-02_task-rsvp_eeg.vhdr"
    eventDescription = 'Event/E  1'
    batch_size = 16
    train_split = 0.8
    dataset = create_dataset(raw_file_path, eventDescription, batch_size)
    
    # Manually shuffle dataset
    dataset = shuffle_dataset(dataset)
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sample = next(iter(data_loader))
    print(len(sample['eeg']))
    data = sample['eeg'][3]
    # plt.plot(data[2, :].cpu().numpy(), label='Original')
    # plt.show()