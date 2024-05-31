import mne
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, random_split

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
        try:
            self.raw.crop(tmax=2000, tmin=1950)
        except:
            self.raw.crop(tmax=1700, tmin=1950)
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
    def extract_epochs_event_paired(self, tmin=-0.128, tmax=0.512):
        print("extracting epochs...")
        self.raw._data = self.raw._data.astype(np.float32)
        events, event_dict = mne.events_from_annotations(self.raw, event_id=None, regexp=self.event_description)
        self.epochs = mne.Epochs(self.raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, preload=True)
        print(self.raw.info) 
    def extract_epochs(self, tmin=-0.128, tmax=0.512, preload=True):
        epoch_length = 0.512
        step = 0.05
        print("extracting epochs...")
        self.raw._data = self.raw._data.astype(np.float32)
        print("Extracting epochs...")
        sfreq = self.raw.info['sfreq']  # Sampling frequency
        epoch_length_samples = int(sfreq * epoch_length)  # Epoch length in samples
        step_samples = int(sfreq * step)  # Step size in samples

        # Calculate number of epochs that can be extracted
        n_epochs = int((len(self.raw.times) - epoch_length_samples) / step_samples) + 1

        # Generate start times for each epoch
        starts = np.arange(0, n_epochs * step_samples, step_samples)

        # Initialize an array to hold the data
        epochs_data = np.zeros((n_epochs, self.raw.get_data().shape[0], epoch_length_samples), dtype=np.float32)

        # Fill the array with epoch data
        for i, start in enumerate(starts):
            end = start + epoch_length_samples
            if end <= len(self.raw.times):
                epochs_data[i] = self.raw._data[:, start:end]

        # Create an MNE Epochs array without needing the events array
        epochs_info = mne.create_info(ch_names=self.raw.info['ch_names'], sfreq=sfreq, ch_types=self.raw.get_channel_types())
        epochs = mne.EpochsArray(epochs_data, epochs_info, tmin=0)

        if preload:
            epochs.load_data()  # Load data into memory if preload is True

        self.epochs = epochs
        
        print(epochs)
        return epochs

    def process_eeg_data(self):
        self.load_raw_data()
        self.preprocess_raw_data()
        self.extract_epochs()
def create_dataset(raw_file_path, event_description, batch_size):
    
    eeg_processor = EEGDataProcessor(raw_file_path, event_description)
    eeg_processor.process_eeg_data()
    dataset = EEGDataLoader(eeg_processor.epochs)
    
    return dataset
class EEGDataLoader:
    def __init__(self, epochs, batch_size=32, val_split=0.1, test_split=0.1):
        eeg_data = epochs.get_data()  # Shape (n_epochs, n_channels, n_times)
        print(eeg_data.shape)
        eeg_data = eeg_data[:, :, :512]  # Adjusting the time dimension to a fixed size

        # Convert to tensor
        eeg_data_tensor = torch.tensor(eeg_data, dtype=torch.float32)

        # Create a dataset
        dataset = TensorDataset(eeg_data_tensor)
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        eeg = self.dataset[index][0]
        return {'eeg': eeg}
        




if __name__ == "__main__":
    raw_file_path = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-04/eeg/sub-04_task-rsvp_eeg.vhdr"
    eventDescription = 'Event/E  1'
    batch_size = 16
    train_split = 0.8
    dataset = create_dataset(raw_file_path, eventDescription, batch_size)
    train_dataloader, val_dataloader, test_dataloader = dataset.get_loaders() 
    sample = next(iter(train_dataloader))
    
    print(len(sample))
    print(f"shape of the sample: {sample[0].shape}")
    data = sample[0]
    plt.plot(data[2, :].cpu().numpy(), label='Original')
    plt.show()
    
    #plot from val_dataloader
    sample = next(iter(val_dataloader))
    data = sample[0]
    plt.plot(data[0, :].cpu().numpy(), label='Original')
    plt.show()
    
    #plot from test_dataloader
    sample = next(iter(test_dataloader))
    data = sample[0]
    plt.plot(data[0, :].cpu().numpy(), label='Original')
    plt.show()
    