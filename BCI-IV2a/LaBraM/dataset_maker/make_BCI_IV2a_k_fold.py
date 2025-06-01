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

from torch.utils.data import Subset
from sklearn.model_selection import KFold

np.random.seed(42)

def convert_and_dump_windows(windows, metadata_df, subfolder, current_subject_id, current_fold_idx, desc="Processing windows"):

    base_path = f"/path/to/data/BCI_IV2a_K_FOLD/Subject_{current_subject_id}/fold_{current_fold_idx}"
    os.makedirs(base_path, exist_ok=True)
    
    folder_path = os.path.join(base_path, subfolder)
    if os.path.exists(folder_path):
        print(f"Folder exists: '{folder_path}'. Deleting the folder and creating a new one...")
        shutil.rmtree(folder_path)
    else:
        print(f"Folder does not exist: '{folder_path}', creating one...")
    
    os.makedirs(folder_path, exist_ok=True)
    
    for idx, window_data in enumerate(tqdm(windows, desc=desc)):
        signal, label, _ = window_data
        
        row = metadata_df.iloc[idx]
        metadata_dict = row.drop("target").to_dict()

        sample = {
            "signal": signal, 
            "label": label, 
            "metadata": metadata_dict
        }
        
        original_window_index = row.name 
        filename = f"subject-{current_subject_id}_fold-{current_fold_idx}_split-{subfolder}_trial-{idx}.pkl"
        filepath = os.path.join(folder_path, filename)

        with open(filepath, "wb") as f:
            pickle.dump(sample, f)

def common_average_reference(data_array_numpy):
    mean_across_channels = data_array_numpy.mean(axis=0, keepdims=True)
    return data_array_numpy - mean_across_channels

preprocessors = [ # maybe change this part
    Preprocessor('pick_types', eeg=True, meg=False, stim=False, eog=False), # only pick EEG channels
    Preprocessor('resample', sfreq=200),
    Preprocessor('filter', l_freq=0.1, h_freq=75.0), # better than 4-38
    Preprocessor('notch_filter', freqs=50.0),
    Preprocessor(common_average_reference),
    Preprocessor(lambda data: np.multiply(data, 1e6)),  # Convert V to micro V
    Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                 factor_new=1e-3, init_block_size=1000)
]

all_subjects = list(range(1, 10))
val_frac = 0.2
k_cv = 3
kf_random_state = 42

if __name__ == "__main__":
    start_time = time.time()
    
    processed_data_subjects = {}
    for subject_id_load in tqdm(all_subjects, desc="Loading and Preprocessing All Subjects"):
        dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id_load])
        preprocess(dataset, preprocessors)

        windows_full = create_windows_from_events(
            dataset,
            trial_start_offset_samples=0,
            trial_stop_offset_samples=0,
            preload=True,
            drop_bad_windows=True,
        )
        metadata_full = windows_full.get_metadata()
        processed_data_subjects[subject_id_load] = (windows_full, metadata_full)

    for subject_id_split in tqdm(all_subjects, desc="Creating K-Fold Splits for Subjects"):
        windows_subject, metadata_subject = processed_data_subjects[subject_id_split]
        
        metadata_subject = metadata_subject.reset_index(drop=True)

        all_window_indices_for_subject = np.arange(len(windows_subject))
        
        kf = KFold(n_splits=k_cv, shuffle=True, random_state=kf_random_state)
        
        for flod_idx, (train_val_indices_kfold, test_indices_kfold) in enumerate(kf.split(all_window_indices_for_subject)):

            current_test_windows = Subset(windows_subject, test_indices_kfold)
            current_metadata_test = metadata_subject.iloc[test_indices_kfold]
            
            num_train_val_fold = len(train_val_indices_kfold)
            
            shuffled_relative_indices_for_train_val = np.random.permutation(num_train_val_fold)
            
            val_size_fold = int(val_frac * num_train_val_fold)
            
            val_relative_indices = shuffled_relative_indices_for_train_val[:val_size_fold]
            train_relative_indices = shuffled_relative_indices_for_train_val[val_size_fold:]
            
            actual_val_indices = train_val_indices_kfold[val_relative_indices]
            actual_train_indices = train_val_indices_kfold[train_relative_indices]
            
            current_train_windows = Subset(windows_subject, actual_train_indices)
            current_metadata_train = metadata_subject.iloc[actual_train_indices]
            
            current_val_windows = Subset(windows_subject, actual_val_indices)
            current_metadata_val = metadata_subject.iloc[actual_val_indices]
            
            desc_prefix = f"S{subject_id_split} F{flod_idx}"
            convert_and_dump_windows(current_train_windows, metadata_df=current_metadata_train, 
                                     subfolder="processed_train", 
                                     current_subject_id=subject_id_split, current_fold_idx=flod_idx,
                                     desc=f"{desc_prefix} Train")
            convert_and_dump_windows(current_val_windows, metadata_df=current_metadata_val,
                                     subfolder="processed_eval", 
                                     current_subject_id=subject_id_split, current_fold_idx=flod_idx,
                                     desc=f"{desc_prefix} Eval")
            convert_and_dump_windows(current_test_windows, metadata_df=current_metadata_test,
                                     subfolder="processed_test", 
                                     current_subject_id=subject_id_split, current_fold_idx=flod_idx,
                                     desc=f"{desc_prefix} Test")
                                     
            print(f"Dumped train/eval/test split for Subject {subject_id_split}, Fold {flod_idx}")
            print(f"  Train size: {len(current_train_windows)}, Val size: {len(current_val_windows)}, Test size: {len(current_test_windows)}")

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")
