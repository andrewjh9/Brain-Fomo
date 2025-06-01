# Based on https://braindecode.org/stable/auto_examples/model_building/plot_bcic_iv_2a_moabb_trial.html#sphx-glr-auto-examples-model-building-plot-bcic-iv-2a-moabb-trial-py
# Based on make_TUEV.py

import os
import numpy as np
import pickle
from tqdm import tqdm
import shutil
import pandas as pd
import time

from moabb.datasets import BNCI2014001
from braindecode.datasets.moabb import MOABBDataset
from braindecode.preprocessing import create_windows_from_events, Preprocessor, preprocess, exponential_moving_standardize

from torch.utils.data import Subset, ConcatDataset

np.random.seed(42)

def convert_and_dump_dataset(windows, metadata_df, filename, test_subject_id, desc="Processing windows"):
    """
    Dumps the windows as pickle files in a folder.

    Arguments:
        windows(iterable): Windows to convert. 
        metadata_df (pd.DataFrame): Metadata for the windows.
        subfolder (str): Folder name where the samples will be saved. 
        test_subject_id (int): Test subject ID for the data. 
        desc (str): Description for the progress bar.  

    """
    # Check if the folder exists, if not create it    
    base_path = f"/path/to/data/BCI_IV2a/Subject_{test_subject_id}"
    os.makedirs(base_path, exist_ok=True)
    
    samples = []
    for idx, window in enumerate(tqdm(windows, desc=desc)):        
        signal, label, _= window # signal shape (n_channels=22, n_samples==1000) 4 sec
        # label = metadata_df.iloc[idx]["target"] # the imagination of movement of the left hand (class 1), 
        #                     # right hand (class 2), both feet (class 3), and tongue (class 4)
        row = metadata_df.iloc[idx]
        metadata = row.drop("target").to_dict() # i_window_in_trial, i_start_in_trial, i_stop_in_trial, subject, session, run 

        sample = {
            "signal": signal, 
            "label": label, 
            "metadata": metadata
        }
        
        samples.append(sample)

    filepath = os.path.join(base_path, f"{filename}_{test_subject_id}.pkl")
    with open(filepath, "wb") as f:
        pickle.dump(samples, f)

def common_average_reference(data_array_numpy):
    mean_across_channels = data_array_numpy.mean(axis=0, keepdims=True)
    return data_array_numpy - mean_across_channels


# Preprocess the dataset 
preprocessors = [ # maybe change this part
    Preprocessor('pick_types', eeg=True, meg=False, stim=False, eog=False), # only pick EEG channels
    Preprocessor('filter', l_freq=0.1, h_freq=75.0), # better than 4-38
    Preprocessor('notch_filter', freqs=50.0),
    Preprocessor(common_average_reference),
    Preprocessor(lambda data: np.multiply(data, 1e6)),  # Convert V to micro V
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=1e-3, init_block_size=1000)
]

all_subjects = list(range(1, 10))
val_frac = 0.2

if __name__ == "__main__":
    start_time = time.time()
    
    processed_data_subjects = {}
    for subject_id in all_subjects:
        # Load and preprocess the BCI Competition IV 2a dataset for each subject
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])
        preprocess(dataset, preprocessors)

        # Create windows from events
        windows = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True,
            drop_bad_windows=True,
        ) # drop_bad_windows=True # drop bad windows?

        # Obtain the metadata
        metadata = windows.get_metadata()
        
        # Save the processed data for each subject
        processed_data_subjects[subject_id] = (windows, metadata)

    for test_subject_id in all_subjects:
        train_windows_list = []
        val_windows_list = []

        metadata_train_list = []
        metadata_val_list = []

        # Process each subject
        for subject_id in all_subjects:
            windows, metadata = processed_data_subjects[subject_id]

            if subject_id == test_subject_id:
                test_windows = windows
                metadata_test = metadata
            else:
                # Split the dataset into train/val
                num_windows = len(windows)
                shuffled_idx = np.random.permutation(num_windows)
                val_size = int(val_frac * num_windows)

                val_idx = shuffled_idx[:val_size]
                train_idx = shuffled_idx[val_size:]

                val_windows_subject = Subset(windows, val_idx)
                train_windows_subject = Subset(windows, train_idx)

                train_windows_list.append(train_windows_subject)
                val_windows_list.append(val_windows_subject)
                
                metadata_train_list.append(metadata.iloc[train_idx])
                metadata_val_list.append(metadata.iloc[val_idx])

        # Concatenate the windows and metadata in train/val
        train_windows = ConcatDataset(train_windows_list)
        val_windows = ConcatDataset(val_windows_list)

        metadata_train = pd.concat(metadata_train_list, ignore_index=True)
        metadata_val = pd.concat(metadata_val_list, ignore_index=True)

        # print(f" Nr. of windows in each split:")
        # print(f"  Train: {len(train_windows)}")
        # print(f"  Val:   {len(val_windows)}")
        # print(f"  Test:  {len(test_windows)}")

        # print(f" Shape of metadata:")
        # print(f"  Train: {metadata_train.shape}")
        # print(f"  Val:   {metadata_val.shape}")
        # print(f"  Test:  {metadata_test.shape}")

        for i, (_, label, _) in enumerate(train_windows):
            expected_label = metadata_train.iloc[i]['target']
            actual_label = label
            assert actual_label == expected_label, f"Mismatch in train at index {i}"
            
        for i, (_, label, _) in enumerate(val_windows):
            expected_label = metadata_val.iloc[i]['target']
            actual_label = label
            assert actual_label == expected_label, f"Mismatch in val at index {i}"
            
        for i, (_, label, _) in enumerate(test_windows):        
            expected_label = metadata_test.iloc[i]['target']
            actual_label = label
            assert actual_label == expected_label, f"Mismatch in test at index {i}"

        # print("\nNumber of windows per subject in each split:")

        # print("\nTrain set:")
        # print(metadata_train['subject'].value_counts().sort_index())

        # print("\nValidation set:")
        # print(metadata_val['subject'].value_counts().sort_index())

        # print("\nTest set:")
        # print(metadata_test['subject'].value_counts().sort_index())

        # Convert and dump the windows as samples
        convert_and_dump_dataset(train_windows, metadata_df=metadata_train, filename="train", test_subject_id=test_subject_id, desc="Processing training windows")
        convert_and_dump_dataset(val_windows, metadata_df=metadata_val, filename="val", test_subject_id=test_subject_id, desc="Processing validation windows")
        convert_and_dump_dataset(test_windows, metadata_df=metadata_test, filename="test", test_subject_id=test_subject_id, desc="Processing test windows")
        print(f"Dumped train/val/test split for test subject {test_subject_id}")
        
    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
