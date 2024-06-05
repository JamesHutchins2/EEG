import mne
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
participant_list = [2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 19, 20,
        21, 22, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 45, 46, 47, 48
    ]

class EEGDataProcessor:
    def __init__(self, raw_file_path, event_description, resample_factor=3):
        pwd = os.getcwd()
        self.raw_file_path = raw_file_path  
        self.event_description = event_description 
        self.resample_factor = resample_factor
        existing_channel_locations = pd.read_csv(pwd + "/electrode_locations/old_points.csv")
        new_channel_locations = pd.read_csv(pwd + "/electrode_locations/new_points.csv")
        
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
            self.raw.crop(tmax=1700, tmin=1650)
        
        
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
def make_path_from_participant_num(participant_index):
    
    
        
    root = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
    #root = "/home/hutchins/projects/def-yalda/hutchins/data/sub-"
    mid = "/eeg/sub-"
    end = "_task-rsvp_eeg.vhdr"
        
    if participant_index < 10:
        path = root + "0" + str(participant_index ) + mid + "0" + str(participant_index) + end
    else:
        path = root + str(participant_index ) + mid + str(participant_index ) + end

    root_event = "/mnt/a/MainFolder/Neural Nirvana/Data/sub-"
    mid = "/eeg/sub-"
    end = "_task-rsvp_events.csv"
    
    if participant_index < 10:
        path_event = root_event + "0" + str(participant_index ) + mid + "0" + str(participant_index) + end
    else:
        path_event = root_event + str(participant_index ) + mid + str(participant_index ) + end
    
    return path, path_event


def get_participant_data(index):
    path, event_path = make_path_from_participant_num(index)
    data_processor = EEGDataProcessor(path, 'Event/E  1')
    data_processor.load_raw_data()
    data_processor.preprocess_raw_data()
    data_processor.extract_epochs()
    
    # get the channel names
    channel_names = data_processor.raw.ch_names
    # get the sampling frequency
    sampling_frequency = data_processor.raw.info['sfreq']
    
    return data_processor, channel_names, sampling_frequency, event_path

class EEGDataLoader:
    def __init__(self, epochs, event_file_name):
        self.event_file_name = event_file_name
        self.events = pd.read_csv(self.event_file_name)
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
        index = int(index)
        eeg = self.dataset[index][0]
        caption = self.events.iloc[index]['caption']
        return {'eeg': eeg, 
                'caption': caption
                }
def load_and_process_data(participant_list):
    """
    Incrementally processes EEG data for all participants.
    Accumulates sums and counts for each caption to compute averages later.
    Returns two dictionaries, one for sums and one for counts.
    """
    sums_by_caption = {}
    counts_by_caption = {}
    
    for participant in participant_list:
        print(f"Processing data for participant {participant}")
        data_processor, channel_names, sampling_frequency, event_file_name = get_participant_data(participant)  # Your existing function
        epochs = data_processor.epochs # Assuming this method exists

        dl = EEGDataLoader(epochs, event_file_name)  # Your existing class
        
        for data in dl:
            caption = data['caption']
            eeg_data = data['eeg']
            
            if caption not in sums_by_caption:
                sums_by_caption[caption] = 0
                counts_by_caption[caption] = 0
            
            sums_by_caption[caption] += eeg_data
            counts_by_caption[caption] += 1
    
    return sums_by_caption, counts_by_caption, channel_names, sampling_frequency

def create_mne_raw(averaged_data, channel_names, sampling_frequency, events_data):
    """
    Create an MNE RawArray object from averaged EEG data and add event annotations.
    
    Parameters:
    averaged_data (dict): A dictionary with captions as keys and averaged EEG data tensors as values.
    channel_names (list): List of EEG channel names.
    sampling_frequency (int): The sampling frequency of the data.
    events_data (list): List of tuples containing event information (caption, onset, duration, event_id).
    
    Returns:
    mne.io.RawArray: The MNE RawArray object with the EEG data.
    """
    # Example: converting a single caption's data for simplicity
    caption = list(averaged_data.keys())[0]
    data = averaged_data[caption].numpy()  # Convert from Tensor to NumPy array, ensure it's 2D: (n_channels, n_times)
    
    # Create MNE info structure
    info = mne.create_info(ch_names=channel_names, sfreq=sampling_frequency, ch_types='eeg')
    
    # Create RawArray
    raw = mne.io.RawArray(data, info)
    
    # Create Annotations
    annotations = mne.Annotations(onset=[], duration=[], description=[])
    for caption, onset, duration, event_id in events_data:
        annotations.append(onset=onset, duration=duration, description=str(event_id))
    
    # Add annotations to raw data
    raw.set_annotations(annotations)
    
    return raw

def save_eeg_data(raw, base_filename):
    """
    Save the RawArray to EEG, VMRK, and VHDR files.
    
    Parameters:
    raw (mne.io.RawArray): The MNE RawArray object with the EEG data.
    base_filename (str): Base path and filename to save the files without extension.
    """
    # Save data
    raw.save(f'{base_filename}.fif', overwrite=True)
    # Export to BrainVision format
    mne.export.export_raw(f'{base_filename}.eeg', raw, fmt='brainvision', overwrite=True)
    
    # Save the events to CSV
    events, event_id = mne.events_from_annotations(raw)
    events_df = pd.DataFrame(events, columns=['time', 'duration', 'event'])
    events_df.to_csv(f'{base_filename}.csv', index=False)

def compute_averages(sums_by_caption, counts_by_caption):
    """
    Compute averages from sums and counts for each caption.
    Returns a dictionary where keys are captions and values are the averaged EEG data.
    """
    averaged_data = {}
    
    for caption in sums_by_caption:
        averaged_data[caption] = sums_by_caption[caption] / counts_by_caption[caption]
    
    return averaged_data

# Example usage
sums_by_caption, counts_by_caption, channel_names, sampling_frequency = load_and_process_data(participant_list)
averaged_data = compute_averages(sums_by_caption, counts_by_caption)

# Define event data: List of tuples (caption, onset, duration, event_id)
# You need to define this according to your data
events_data = [
    ("caption1", 10.0, 0.5, 1), 
    ("caption2", 20.0, 0.5, 2),
    # Add more events as needed
]

raw = create_mne_raw(averaged_data, channel_names, sampling_frequency, events_data)

# Save the data
save_eeg_data(raw, '/mnt/a/MainFolder/Neural Nirvana/encoder_transformer/model_copy/EEG/GESM/src/dataloader/avg_data')
    


