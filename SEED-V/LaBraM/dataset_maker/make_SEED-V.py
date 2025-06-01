import mne
import numpy as np
import os
import pickle
from tqdm import tqdm
import re

chOrder_standard = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1",
    "CZ", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ",
    "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"
]

all_segments_info = [
    {"session": 1, "start": 30, "end": 102, "label_text": "Happy", "trial_in_session": 1},
    {"session": 1, "start": 132, "end": 228, "label_text": "Fear", "trial_in_session": 2},
    {"session": 1, "start": 287, "end": 524, "label_text": "Neutral", "trial_in_session": 3},
    {"session": 1, "start": 555, "end": 742, "label_text": "Sad", "trial_in_session": 4},
    {"session": 1, "start": 773, "end": 920, "label_text": "Disgust", "trial_in_session": 5},
    {"session": 1, "start": 982, "end": 1240, "label_text": "Happy", "trial_in_session": 6},
    {"session": 1, "start": 1271, "end": 1568, "label_text": "Fear", "trial_in_session": 7},
    {"session": 1, "start": 1628, "end": 1697, "label_text": "Neutral", "trial_in_session": 8},
    {"session": 1, "start": 1730, "end": 1994, "label_text": "Sad", "trial_in_session": 9},
    {"session": 1, "start": 2025, "end": 2166, "label_text": "Disgust", "trial_in_session": 10},
    {"session": 1, "start": 2227, "end": 2401, "label_text": "Happy", "trial_in_session": 11},
    {"session": 1, "start": 2435, "end": 2607, "label_text": "Fear", "trial_in_session": 12},
    {"session": 1, "start": 2667, "end": 2901, "label_text": "Neutral", "trial_in_session": 13},
    {"session": 1, "start": 2932, "end": 3172, "label_text": "Sad", "trial_in_session": 14},
    {"session": 1, "start": 3204, "end": 3359, "label_text": "Disgust", "trial_in_session": 15},

    {"session": 2, "start": 30, "end": 267, "label_text": "Sad", "trial_in_session": 1},
    {"session": 2, "start": 299, "end": 488, "label_text": "Fear", "trial_in_session": 2},
    {"session": 2, "start": 548, "end": 614, "label_text": "Neutral", "trial_in_session": 3},
    {"session": 2, "start": 646, "end": 773, "label_text": "Disgust", "trial_in_session": 4},
    {"session": 2, "start": 836, "end": 967, "label_text": "Happy", "trial_in_session": 5},
    {"session": 2, "start": 1000, "end": 1059, "label_text": "Happy", "trial_in_session": 6},
    {"session": 2, "start": 1091, "end": 1331, "label_text": "Disgust", "trial_in_session": 7},
    {"session": 2, "start": 1392, "end": 1622, "label_text": "Neutral", "trial_in_session": 8},
    {"session": 2, "start": 1657, "end": 1777, "label_text": "Sad", "trial_in_session": 9},
    {"session": 2, "start": 1809, "end": 1908, "label_text": "Fear", "trial_in_session": 10},
    {"session": 2, "start": 1966, "end": 2153, "label_text": "Neutral", "trial_in_session": 11},
    {"session": 2, "start": 2186, "end": 2302, "label_text": "Happy", "trial_in_session": 12},
    {"session": 2, "start": 2333, "end": 2428, "label_text": "Fear", "trial_in_session": 13},
    {"session": 2, "start": 2490, "end": 2709, "label_text": "Sad", "trial_in_session": 14},
    {"session": 2, "start": 2741, "end": 2817, "label_text": "Disgust", "trial_in_session": 15},

    {"session": 3, "start": 30, "end": 321, "label_text": "Sad", "trial_in_session": 1},
    {"session": 3, "start": 353, "end": 418, "label_text": "Fear", "trial_in_session": 2},
    {"session": 3, "start": 478, "end": 643, "label_text": "Neutral", "trial_in_session": 3},
    {"session": 3, "start": 674, "end": 764, "label_text": "Disgust", "trial_in_session": 4},
    {"session": 3, "start": 825, "end": 877, "label_text": "Happy", "trial_in_session": 5},
    {"session": 3, "start": 908, "end": 1147, "label_text": "Happy", "trial_in_session": 6},
    {"session": 3, "start": 1200, "end": 1284, "label_text": "Disgust", "trial_in_session": 7},
    {"session": 3, "start": 1346, "end": 1418, "label_text": "Neutral", "trial_in_session": 8},
    {"session": 3, "start": 1451, "end": 1679, "label_text": "Sad", "trial_in_session": 9},
    {"session": 3, "start": 1711, "end": 1996, "label_text": "Fear", "trial_in_session": 10},
    {"session": 3, "start": 2055, "end": 2275, "label_text": "Neutral", "trial_in_session": 11},
    {"session": 3, "start": 2307, "end": 2425, "label_text": "Happy", "trial_in_session": 12},
    {"session": 3, "start": 2457, "end": 2664, "label_text": "Fear", "trial_in_session": 13},
    {"session": 3, "start": 2726, "end": 2857, "label_text": "Sad", "trial_in_session": 14},
    {"session": 3, "start": 2888, "end": 3066, "label_text": "Disgust", "trial_in_session": 15}
]

label_map = {
    "Disgust": 0,
    "Fear": 1,
    "Sad": 2,
    "Neutral": 3,
    "Happy": 4
}

fs = 200.0

def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

INPUT_PATH = "home/scur0546/original_code/data/SEED-V/"
OUTPUT_PATH = "scratch-shared/scur0546/data/SEED-V"

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
    print(f"Created folder: {OUTPUT_PATH}")

OUTPUT_PATH_PROCESSED_TRAIN = os.path.join(OUTPUT_PATH, "processed_train")
OUTPUT_PATH_PROCESSED_EVAL = os.path.join(OUTPUT_PATH, "processed_eval")
OUTPUT_PATH_PROCESSED_TEST = os.path.join(OUTPUT_PATH, "processed_test")

for path in [OUTPUT_PATH_PROCESSED_TRAIN, OUTPUT_PATH_PROCESSED_EVAL, OUTPUT_PATH_PROCESSED_TEST]:
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder exists: {path}")
print("All output folders checked and ready!\n")

with open(os.path.join(OUTPUT_PATH, "seedv_template_processing_errors.txt"), "w") as f:
    f.write("SEED-V Template Processing Errors Log:\n")

input_eeg_raw_dir = os.path.join(INPUT_PATH, "EEG_raw")

file_pattern = re.compile(r"(\d+)_(\d+)_(\d+)\.cnt")
cnt_files = [f for f in os.listdir(input_eeg_raw_dir) if f.endswith(".cnt")]

for eeg_filename in tqdm(cnt_files, desc="Processing EEG files"):
    match = file_pattern.match(eeg_filename)

    subject_id_str = match.group(1)
    session_id_from_file = int(match.group(2))
    cnt_filepath = os.path.join(input_eeg_raw_dir, eeg_filename)

    Rawdata = mne.io.read_raw_cnt(cnt_filepath, preload=True, verbose=False)
    
    current_raw_channels = Rawdata.ch_names
    channels_to_pick = [ch for ch in chOrder_standard if ch in current_raw_channels]
    Rawdata.pick_channels(channels_to_pick, ordered=False)
    Rawdata.reorder_channels(channels_to_pick)

    Rawdata.filter(l_freq=0.1, h_freq=75.0)
    Rawdata.notch_filter(50.0)
    Rawdata.resample(200, n_jobs=5)

    signals_processed = Rawdata.get_data(units='uV')
    segments_for_current_session = [seg for seg in all_segments_info if seg["session"] == session_id_from_file]

    for segment in segments_for_current_session:
        start_sec = segment["start"]
        end_sec = segment["end"]
        label_text = segment["label_text"]
        trial_in_session = segment["trial_in_session"]

        emotion_code = label_map.get(label_text)

        start_sample_orig = int(start_sec * 1000.0)
        end_sample_orig = int(end_sec * 1000.0)

        start_sample_new = int(start_sample_orig * (fs / 1000.0))
        end_sample_new = int(end_sample_orig * (fs / 1000.0))

        segment_signal_data = signals_processed[:, start_sample_new:end_sample_new]

        samples_per_epoch = int(fs)
        num_epochs_in_segment = segment_signal_data.shape[1] // samples_per_epoch

        for epoch_k in range(num_epochs_in_segment):
            epoch_data = segment_signal_data[:, epoch_k * samples_per_epoch : (epoch_k + 1) * samples_per_epoch]

            if epoch_data.shape[1] == samples_per_epoch:
                if 1 <= trial_in_session <= 5:
                    output_dir_for_epoch = OUTPUT_PATH_PROCESSED_TRAIN
                elif 6 <= trial_in_session <= 10:
                    output_dir_for_epoch = OUTPUT_PATH_PROCESSED_EVAL
                elif 11 <= trial_in_session <= 15:
                    output_dir_for_epoch = OUTPUT_PATH_PROCESSED_TEST
                else:
                    continue
                
                epoch_savename = f"s{subject_id_str}_sess{session_id_from_file}_t{trial_in_session:02d}_e{epoch_k:03d}.pkl"
                dump_path = os.path.join(output_dir_for_epoch, epoch_savename)
                
                data_to_save = {"X": epoch_data, "y": emotion_code}
                save_pickle(data_to_save, dump_path)
                
# CODE BELOW IS NOT NECESSARY, ITS A DIFFERENT TRAIN / VAL / TEST SPLIT. NOT 5-5-5 subjects
# -----------------------------------------------------------------------------------------s

# # transfer to train, eval, and test
# root = OUTPUT_PATH
# seed = 4523
# np.random.seed(seed)

# train_files = os.listdir(OUTPUT_PATH_PROCESSED_TRAIN)
# train_sub = list(set([f.split("_")[0] for f in train_files]))
# print("train sub", len(train_sub))
# test_files = os.listdir(OUTPUT_PATH_PROCESSED_TEST)

# val_sub = np.random.choice(train_sub, size=int(
#     len(train_sub) * 0.2), replace=False)
# train_sub = list(set(train_sub) - set(val_sub))
# val_files = [f for f in train_files if f.split("_")[0] in val_sub]
# train_files = [f for f in train_files if f.split("_")[0] in train_sub]

# os.makedirs(os.path.join(root, 'processed', 'processed_train'), exist_ok=True)
# os.makedirs(os.path.join(root, 'processed', 'processed_eval'), exist_ok=True)
# os.makedirs(os.path.join(root, 'processed', 'processed_test'), exist_ok=True)

# for file in train_files:
#     os.system(f"cp {os.path.join(OUTPUT_PATH_PROCESSED_TRAIN, file)} {os.path.join(root, 'processed', 'processed_train')}")
# for file in val_files:
#     os.system(f"cp {os.path.join(OUTPUT_PATH_PROCESSED_TRAIN, file)} {os.path.join(root, 'processed', 'processed_eval')}")
# for file in test_files:
#     os.system(f"cp {os.path.join(OUTPUT_PATH_PROCESSED_TEST, file)} {os.path.join(root, 'processed', 'processed_test')}")
