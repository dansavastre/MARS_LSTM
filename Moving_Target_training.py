import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from utils import create_sequences_rsf
from utils import define_LSTM
from utils import define_CNN
from utils import training_loop


# Define parameters
model_name = f"AST_MARS"
seq_length = 16
step = 5
epochs = 150
batch_size = 128
n_keypoints = 57
n_iterations = 10

results_list = []


# Load Moving Target dataset
featuremap_train = np.load(f'Asterios Dataset/mmWave/0/training_mmWave.npy')
featuremap_validate = np.load(f'Asterios Dataset/mmWave/0/validate_mmWave.npy')
featuremap_test = np.load(f'Asterios Dataset/mmWave/0/testing_mmWave.npy')

labels_train = np.load(f'Asterios Dataset/kinect/0/training_labels.npy')
labels_validate = np.load(f'Asterios Dataset/kinect/0/validate_labels.npy')
labels_test = np.load(f'Asterios Dataset/kinect/0/testing_labels.npy')

# Create sequences
# X_train, y_train       = create_sequences_rsf(featuremap_train, labels_train, seq_length, step=step)
# X_validate, y_validate = create_sequences_rsf(featuremap_validate, labels_validate, seq_length, step=step)
# X_test, y_test         = create_sequences_rsf(featuremap_test, labels_test, seq_length, step=step)

# model = define_LSTM(X_train[0].shape, n_keypoints, n_keypoints)
# model = define_CNN(X_train[0].shape, n_keypoints)

# Train the model
# training_loop(model, model_name, X_train, y_train, X_validate, y_validate, X_test, y_test, batch_size=batch_size, epochs=epochs, n_iterations=n_iterations)

model = define_CNN(featuremap_train[0].shape, n_keypoints)
training_loop(model, model_name, featuremap_train, labels_train, featuremap_validate, labels_validate, 
              featuremap_test, labels_test, batch_size=batch_size, epochs=epochs, n_iterations=n_iterations)