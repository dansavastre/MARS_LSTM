# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:59:44 2021

@author: xxx
"""

"""
import all the necessary packages
Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from sklearn import metrics
import datetime

from keras.optimizers import Adam
from keras.models import Model
from keras.models import Sequential
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import LSTM, Reshape
from keras.callbacks import CSVLogger

from utils import create_sequences_rsf
from utils import create_sequences_rst

# set the directory
import os
path = os.getcwd()
os.chdir(path)


#load the feature and labels, 24066, 8033, and 7984 frames for train, validate, and test
featuremap_train = np.load('feature/featuremap_train.npy')
featuremap_validate = np.load('feature/featuremap_validate.npy')
featuremap_test = np.load('feature/featuremap_test.npy')

labels_train = np.load('feature/labels_train.npy')
labels_validate = np.load('feature/labels_validate.npy')
labels_test = np.load('feature/labels_test.npy')


# Define Global Variables
paper_result_list = []
base_name = 'AST_LSTM_rsf'
seq_length = 16
step = 5
batch_size = 128
epochs = 150
n_keypoints = 57



def define_LSTM_CNN(input_shape, n_keypoints):
    model = Sequential()

    # Define the input layer
    model.add(Input(shape=input_shape))
    
    # CNN layers
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')))
    model.add(Dropout(0.3))
    model.add(BatchNormalization(momentum=0.95))
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512, activation='relu')))

    model.add(BatchNormalization(momentum=0.95))
    model.add(Dropout(0.4))
    
    # LSTM layer
    model.add(LSTM(units=n_keypoints, return_sequences=False))

    # Output layer
    model.add(Dense(n_keypoints, activation='linear'))

    # compile the model
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    
    return model

def training_loop(model, model_name, featuremap_train, labels_train, featuremap_validate, labels_validate, featuremap_test, labels_test, batch_size=128, epochs=150, num_iterations=10):
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Repeat i iteration to get the average result
    for i in range(num_iterations):
        print('Iteration', i)

        # instantiate the model
        keypoint_model = model
            
        # initial maximum error 
        score_min = 10
        csv_logger = CSVLogger(f'logs/{model_name}_tain.log')
        history = keypoint_model.fit(featuremap_train, labels_train,
                                    batch_size=batch_size, epochs=epochs, verbose=1, 
                                    validation_data=(featuremap_validate, labels_validate),
                                    callbacks=[tensorboard_callback, csv_logger])
        
        print(keypoint_model.summary())

        # save and print the metrics
        score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
        print('train MAPE = ', score_train[3])

        score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
        print('test MAPE = ', score_test[3])

        print('Score for test:', score_test)
        
        result_test = keypoint_model.predict(featuremap_test)

        # # Plot accuracy
        # plt.plot(history.history['mae'])
        # plt.plot(history.history['val_mae'])
        # plt.title('Model accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Xval'], loc='upper left')
        # plt.savefig(f'plots/accuracy_{model_name}_{i}.png')  # Save the figure with iteration number
        # plt.close()  # Close the figure

        # # Plot loss
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Xval'], loc='upper left')
        # plt.xlim([0,100])
        # plt.ylim([0,0.1])
        # plt.savefig(f'plots/loss_{model_name}_{i}.png')  # Save the figure with iteration number
        # plt.close()  # Close the figure

        
        
        # For using return_sequences = True, the output is 3D, we need to reshape it to 2D
        labels_test_reshaped = labels_test.reshape(-1, 57)
        result_test_reshaped = result_test.reshape(-1, 57)

        # error for each axis
        print("mae for x is",metrics.mean_absolute_error(labels_test_reshaped[:,0:19], result_test_reshaped[:,0:19]))
        print("mae for y is",metrics.mean_absolute_error(labels_test_reshaped[:,19:38], result_test_reshaped[:,19:38]))
        print("mae for z is",metrics.mean_absolute_error(labels_test_reshaped[:,38:57], result_test_reshaped[:,38:57]))
        
        # matrix transformation for the final all 19 points mae
        x_mae = metrics.mean_absolute_error(labels_test_reshaped[:,0:19], result_test_reshaped[:,0:19], multioutput = 'raw_values')
        y_mae = metrics.mean_absolute_error(labels_test_reshaped[:,19:38], result_test_reshaped[:,19:38], multioutput = 'raw_values')
        z_mae = metrics.mean_absolute_error(labels_test_reshaped[:,38:57], result_test_reshaped[:,38:57], multioutput = 'raw_values')
        
        all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3,19)
        avg_19_points_mae = np.mean(all_19_points_mae, axis = 0)
        avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis = 1).reshape(1,3)

        all_19_points_mae_Transpose = all_19_points_mae.T
        
        # matrix transformation for the final all 19 points rmse
        x_rmse = metrics.mean_squared_error(labels_test_reshaped[:,0:19], result_test_reshaped[:,0:19], multioutput = 'raw_values', squared=False)
        y_rmse = metrics.mean_squared_error(labels_test_reshaped[:,19:38], result_test_reshaped[:,19:38], multioutput = 'raw_values', squared=False)
        z_rmse = metrics.mean_squared_error(labels_test_reshaped[:,38:57], result_test_reshaped[:,38:57], multioutput = 'raw_values', squared=False)
        
        all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3,19)
        avg_19_points_rmse = np.mean(all_19_points_rmse, axis = 0)
        avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis = 1).reshape(1,3)

        all_19_points_rmse_Transpose = all_19_points_rmse.T
        
        # merge the mae and rmse
        all_19_points_maermse_Transpose = np.concatenate((all_19_points_mae_Transpose,all_19_points_rmse_Transpose), axis = 1)*100
        avg_19_points_maermse_Transpose = np.concatenate((avg_19_points_mae_xyz,avg_19_points_rmse_xyz), axis = 1)*100
        
        # concatenate the array, the final format is the same as shown in paper. First 19 rows each joint, the final row is the average
        paper_result_maermse = np.concatenate((all_19_points_maermse_Transpose, avg_19_points_maermse_Transpose), axis = 0)
        paper_result_maermse = np.around(paper_result_maermse, 2)
        # reorder the columns to make it xmae, xrmse, ymae, yrmse, zmae, zrmse, avgmae, avgrmse
        paper_result_maermse = paper_result_maermse[:, [0,3,1,4,2,5]]

        # append each iterations result
        paper_result_list.append(paper_result_maermse)
        
        #define the output directory
        output_direct = 'models/'
        
        if not os.path.exists(output_direct):
            os.makedirs(output_direct)

        # save the best model so far
        if(score_test[1] < score_min):
            # keypoint_model.save(output_direct + 'MARS_LSTM.h5')
            keypoint_model.save(output_direct + f'{model_name}.keras')
            score_min = score_test[1]


    # average the result for all iterations
    mean_paper_result_list = np.mean(paper_result_list, axis = 0)
    mean_mae = np.mean(np.dstack((mean_paper_result_list[:,0], mean_paper_result_list[:,2], mean_paper_result_list[:,4])).reshape(20,3), axis = 1)
    mean_rmse = np.mean(np.dstack((mean_paper_result_list[:,1], mean_paper_result_list[:,3], mean_paper_result_list[:,5])).reshape(20,3), axis = 1)
    mean_paper_result_list = np.concatenate((np.mean(paper_result_list, axis = 0), mean_mae.reshape(20,1), mean_rmse.reshape(20,1)), axis = 1)

    #Export the Accuracy
    output_path = output_direct + "Accuracy"
    output_filename = output_path + f'/{model_name}_accuracy'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.save(output_filename + ".npy", mean_paper_result_list)
    np.savetxt(output_filename + ".txt", mean_paper_result_list,fmt='%.2f')

    plt.show()



if __name__ == '__main__':
    
    for i in range(10):

        # Load Asterios dataset
        featuremap_train = np.load(f"Asterios Dataset/mmWave/{i}/training_mmWave.npy")
        featuremap_validate = np.load(f"Asterios Dataset/mmWave/{i}/validate_mmWave.npy")
        featuremap_test = np.load(f"Asterios Dataset/mmWave/{i}/testing_mmWave.npy")

        labels_train = np.load(f"Asterios Dataset/kinect/{i}/training_labels.npy")
        labels_validate = np.load(f"Asterios Dataset/kinect/{i}/validate_labels.npy")
        labels_test = np.load(f"Asterios Dataset/kinect/{i}/testing_labels.npy")

        # Create sequences
        X_train, y_train       = create_sequences_rsf(featuremap_train, labels_train, seq_length, step=step)
        X_validate, y_validate = create_sequences_rsf(featuremap_validate, labels_validate, seq_length, step=step)
        X_test, y_test         = create_sequences_rsf(featuremap_test, labels_test, seq_length, step=step)

        # Define the model
        model = define_LSTM_CNN(X_train[0].shape, n_keypoints)
        model_name = base_name + '_' + str(seq_length)

        # Train the model
        training_loop(model, model_name, X_train, y_train, 
                    X_validate, y_validate, X_test, y_test, 
                    batch_size=batch_size, epochs=epochs, num_iterations=10)
