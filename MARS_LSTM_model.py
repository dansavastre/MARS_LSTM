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
# Initialize the result array
paper_result_list = []
# define batch size and epochs
batch_size = 128
epochs = 150
# define model name
model_name = 'LSTM'


def create_sequence(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length // 2])
    
    return np.array(X), np.array(y)

def define_LSTM_CNN(input_shape, n_keypoints):
    model = Sequential()
    
    # CNN layers
    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=input_shape)))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation='relu')))
    model.add(TimeDistributed(Dropout(0.3)))
    model.add(TimeDistributed(BatchNormalization(momentum=0.95)))
    
    # Reshape output from CNN layer to fit LSTM layer input shape
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layers
    model.add(LSTM(units=128, return_sequences=False))

    # Fully connected layer
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization(momentum=0.95))
    model.add(Dropout(0.4))
    
    # Output layer
    model.add(Dense(n_keypoints, activation='linear'))
    
    # compile the model
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    
    return model


def training_loop(model, model_name, batch_size, epochs, featuremap_train, labels_train, featuremap_validate, labels_validate, featuremap_test, labels_test):

    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Repeat i iteration to get the average result
    for i in range(10):
        print('Iteration', i)
        # instantiate the model
        keypoint_model = model
            
        # initial maximum error 
        score_min = 10
        history = keypoint_model.fit(featuremap_train, labels_train,
                                    batch_size=batch_size, epochs=epochs, verbose=1, 
                                    validation_data=(featuremap_validate, labels_validate))
        
        # save and print the metrics
        score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
        print('train MAPE = ', score_train[3])

        score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
        print('test MAPE = ', score_test[3])
        
        result_test = keypoint_model.predict(featuremap_test)

        # Plot accuracy
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Xval'], loc='upper left')
        plt.savefig(f'plots/accuracy_{model_name}_{i}.png')  # Save the figure with iteration number
        plt.close()  # Close the figure

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Xval'], loc='upper left')
        plt.xlim([0,100])
        plt.ylim([0,0.1])
        plt.savefig(f'plots/loss_{model_name}_{i}.png')  # Save the figure with iteration number
        plt.close()  # Close the figure

        
        


        # error for each axis
        print("mae for x is",metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19]))
        print("mae for y is",metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38]))
        print("mae for z is",metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57]))
        
        # matrix transformation for the final all 19 points mae
        x_mae = metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values')
        y_mae = metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values')
        z_mae = metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values')
        
        all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3,19)
        avg_19_points_mae = np.mean(all_19_points_mae, axis = 0)
        avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis = 1).reshape(1,3)

        all_19_points_mae_Transpose = all_19_points_mae.T
        
        # matrix transformation for the final all 19 points rmse
        x_rmse = metrics.mean_squared_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values', squared=False)
        y_rmse = metrics.mean_squared_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values', squared=False)
        z_rmse = metrics.mean_squared_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values', squared=False)
        
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
    X_train, y_train = create_sequence(featuremap_train, labels_train, 10)
    X_validate, y_validate = create_sequence(featuremap_validate, labels_validate, 10)
    X_test, y_test = create_sequence(featuremap_test, labels_test, 10)

    model = define_LSTM_CNN(X_train[0].shape, 57)

    training_loop(model, model_name, batch_size, epochs,
                  X_train, y_train, X_validate, y_validate, X_test, y_test)
