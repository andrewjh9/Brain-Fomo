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

for test_subject in range(1, n_subjects + 1):
    print(f"=== Subject {test_subject} ===")
    
    # Load the dataset for the current subject
    data_dir = '/path/to/data/BCI_IV2a/'
    train_path = os.path.join(data_dir, f'Subject_{test_subject}', f'train_{test_subject}.pkl')
    val_path = os.path.join(data_dir, f'Subject_{test_subject}', f'val_{test_subject}.pkl')
    test_path = os.path.join(data_dir, f'Subject_{test_subject}', f'test_{test_subject}.pkl')
    
    X_train, y_train = load_dataset(train_path) 
    X_val, y_val = load_dataset(val_path)
    X_test, y_test = load_dataset(test_path)

    # Define the EEGNet-8,2,16 model
    model = EEGNet(nb_classes=4, Chans=22, Samples=1000,
                   dropoutRate=0.5, kernLength=125, F1=8, D=2, F2=16,
                   dropoutType='Dropout')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Set a path for the checkpoints
    checkpoint_dir = '/path/to/checkpoints_LOSO/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_subject_{test_subject}.h5')
    checkpointer = ModelCheckpoint(filepath=checkpoint_path, verbose=2, save_best_only=True)

    # Fit the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, batch_size=64, epochs=50, verbose=2,
              validation_data=(X_val, y_val), callbacks=[early_stopping, checkpointer])

    # Load the best model
    model.load_weights(checkpoint_path)

    # Evaluate the model on the test set
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Accuracy Test Subject {test_subject}: {acc:.4f}")
    accuracies.append(acc)

mean_acc = np.mean(accuracies)
print(f"\n=== LOSO Mean Accuracy from {n_subjects}: {mean_acc:.4f}===")
