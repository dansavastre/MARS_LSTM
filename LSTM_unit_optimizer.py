import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils import create_sequences_rsf
from utils import define_LSTM
from utils import training_loop


# Define parameters
seq_length = 16
step = 5
epochs = 100
batch_size = 128
n_keypoints = 57
n_iterations = 5

# Values to be tested
n_units_vals = [32, 57, 64, 128, 256]



#load the feature and labels, 24066, 8033, and 7984 frames for train, validate, and test
featuremap_train = np.load('feature/featuremap_train.npy')
featuremap_validate = np.load('feature/featuremap_validate.npy')
featuremap_test = np.load('feature/featuremap_test.npy')

labels_train = np.load('feature/labels_train.npy')
labels_validate = np.load('feature/labels_validate.npy')
labels_test = np.load('feature/labels_test.npy')

# Create sequences
X_train, y_train       = create_sequences_rsf(featuremap_train, labels_train, seq_length, step=step)
X_validate, y_validate = create_sequences_rsf(featuremap_validate, labels_validate, seq_length, step=step)
X_test, y_test         = create_sequences_rsf(featuremap_test, labels_test, seq_length, step=step)

for n_units in n_units_vals:
    model = define_LSTM(X_train[0].shape, n_keypoints, n_units)
    model_name = f"Opt_LSTM_rsf_16_{n_units}.keras"


    print(f"\nTraining model {model_name}\n")
    training_loop(model, model_name, X_train, y_train, X_validate, y_validate, X_test, y_test, batch_size=batch_size, epochs=epochs, n_iterations=n_iterations)
