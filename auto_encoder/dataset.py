import mne
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


"""Currenlty cropping the dataset for testing in line 74"""

# Set seed
torch.manual_seed(42)

class EEGDataProcessor:
    def __init__(self, raw_file_path, event_description, resample_factor=3):
        self.raw_file_path = raw_file_path  
        self.event_description = event_description 
        self.resample_factor = resample_factor
        existing_channel_locations = pd.read_csv("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/text_alignment/electrode_locations/old_points.csv")
        new_channel_locations = pd.read_csv("/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/text_alignment/electrode_locations/new_points.csv")
        
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
        print("extracting epochs...")
        self.raw._data = self.raw._data.astype(np.float32)
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=self.event_description)
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)
        print(self.raw.info)

    def process_eeg_data(self):
        self.load_raw_data()
        self.preprocess_raw_data()
        self.extract_epochs()

class EEGDataLoader:
    def __init__(self, epochs):
        eeg_data = epochs.get_data()  # Shape (n_epochs, n_channels, n_times)
        print(eeg_data.shape)
        eeg_data = eeg_data[:, :, :512]
        self.data_len  = 512

        self.n_channels, self.n_times = eeg_data.shape[1], eeg_data.shape[2]
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        self.dataset = TensorDataset(eeg_data_tensor)
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        eeg = self.dataset[index][0]
        return {'eeg': eeg}
        

def create_dataset(raw_file_path, event_description, batch_size):
    
    eeg_processor = EEGDataProcessor(raw_file_path, event_description)
    eeg_processor.process_eeg_data()
    dataset = EEGDataLoader(eeg_processor.epochs)
    
    return dataset


if __name__ == "__main__":
    raw_file_path = "data/sub-04/eeg/sub-04_task-rsvp_eeg.vhdr"
    eventDescription = 'Event/E  1'
    batch_size = 16
    train_split = 0.8
    dataset = create_dataset(raw_file_path, eventDescription, batch_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    sample = next(iter(data_loader))
    print(len(sample['eeg']))
    data = sample['eeg'][3]
    plt.plot(data[2, :].cpu().numpy(), label='Original')
    plt.show()