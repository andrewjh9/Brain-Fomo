# Inspired by https://github.com/vlawhern/arl-eegmodels

import numpy as np
import pickle
import random
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from EEGModels import EEGNet
from tensorflow.keras import backend as K
import os

from sklearn.model_selection import KFold, train_test_split

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

K.set_image_data_format('channels_last')

def load_dataset(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    signals = []
    labels = []
    for sample in data:
        signal = sample["signal"]  # shape: (1, 22, 1000)
        signals.append(signal) 
        labels.append(sample["label"])
    X = np.array(signals)
    X = X[..., np.newaxis] # shape: (n_samples, 22, 1000, 1)
    y = np.array(labels)
    y = to_categorical(y, num_classes=4) # convert to one-hot encoding
    return X, y


n_subjects = 9
accuracies = []
n_folds = 3

for subject_id in range(1, n_subjects + 1):
    print(f"=== Subject {subject_id} ===")
    
    data_dir = '/path/to/data/BCI_IV2a/'
    subject_data_path = os.path.join(data_dir, f'Subject_{subject_id}', f'test_{subject_id}.pkl')
    
    X_all_subject, y_all_subject = load_dataset(subject_data_path)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    subject_fold_accuracies = []
    
    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(X_all_subject)): 
        print(f"--- Fodl {fold_idx + 1}/{n_folds} ---")
        
        X_train_val, X_test_fold = X_all_subject[train_val_indices], X_all_subject[test_indices]
        y_train_val, y_test_fold = y_all_subject[train_val_indices], y_all_subject[test_indices]
        
        y_train_val_labels = np.argmax(y_train_val, axis=1)
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val_labels
        )
        
        model = EEGNet(nb_classes=4, Chans=22, Samples=1000,
                       dropoutRate=0.5, kernLength=125, F1=8, D=2, F2=16,
                       dropoutType='Dropout')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        checkpoint_dir_within_subject = '/path/to/checkpoints_within_subject/'
        os.makedirs(checkpoint_dir_within_subject, exist_ok=True)
        checkpoint_path_fold = os.path.join(checkpoint_dir_within_subject, f'model_subject_{subject_id}_fold_{fold_idx}.h5')
        
        checkpointer = ModelCheckpoint(filepath=checkpoint_path_fold, verbose=0, save_best_only=True, monitor='val_loss') 
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train_fold, y_train_fold, batch_size=16, epochs=50, verbose=2,
                  validation_data=(X_val_fold, y_val_fold), callbacks=[early_stopping, checkpointer])

        model.load_weights(checkpoint_path_fold)

        loss, acc = model.evaluate(X_test_fold, y_test_fold, verbose=0)
        print(f"Fold {fold_idx + 1} Test Acucracy: {acc:.4f}")
        subject_fold_accuracies.append(acc)
    
    mean_acc_subject = np.mean(subject_fold_accuracies)
    print(f"Meean Accuracy for Subject {subject_id} across {n_folds} folds: {mean_acc_subject:.4f}")
    accuracies.append(mean_acc_subject)

mean_acc = np.mean(accuracies)
print(f"\n=== Within-Subject {n_folds}-Fold CV Mean Accuracy across {n_subjects} subjects: {mean_acc:.4f} ===")
