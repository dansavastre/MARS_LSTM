import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, LSTM, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# Function to split data into sequences
# return: For each sequence, the expected prediction is for each frame in the sequence
def create_sequences_rst(data, labels, seq_length, step=1):
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i + seq_length])
        y.append(labels[i:i + seq_length])

    # Pad or truncate sequences to ensure consistent length
    if len(X[-1]) < seq_length:
        remaining = seq_length - len(X[-1])
        X[-1] = np.pad(X[-1], ((0, remaining), (0, 0), (0, 0), (0, 0), (0, 0)), 'constant')
        y[-1] = np.pad(y[-1], ((0, remaining), (0, 0)), 'constant')
    elif len(X[-1]) > seq_length:
        X[-1] = X[-1][:seq_length]
        y[-1] = y[-1][:seq_length]

    return np.array(X), np.array(y)


# Function to split data into sequences
# return: For each sequence, the expected prediction is the last frame in the sequence
def create_sequences_rsf(data, labels, seq_length, step=1):
    X, y = [], []
    for i in range(0, len(data) - seq_length, step):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length - 1])

    return np.array(X), np.array(y)


def calculate_accuracy(y, predictions):
    # matrix transformation for the final all 19 points mae
    x_mae = metrics.mean_absolute_error(y[:,0:19], predictions[:,0:19], multioutput = 'raw_values')
    y_mae = metrics.mean_absolute_error(y[:,19:38], predictions[:,19:38], multioutput = 'raw_values')
    z_mae = metrics.mean_absolute_error(y[:,38:57], predictions[:,38:57], multioutput = 'raw_values')

    all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 19)
    avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis=1).reshape(1, 3)   
    all_19_points_mae_Transpose = all_19_points_mae.T
    # avg_19_points_mae = np.mean(all_19_points_mae, axis=1).reshape(1, 3)


    # matrix transformation for the final all 19 points rmse
    x_rmse = metrics.mean_squared_error(y[:,0:19], predictions[:,0:19], multioutput = 'raw_values', squared=False)
    y_rmse = metrics.mean_squared_error(y[:,19:38], predictions[:,19:38], multioutput = 'raw_values', squared=False)
    z_rmse = metrics.mean_squared_error(y[:,38:57], predictions[:,38:57], multioutput = 'raw_values', squared=False)

    all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3, 19)
    avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis=1).reshape(1, 3)
    all_19_points_rmse_Transpose = all_19_points_rmse.T
    # avg_19_points_rmse = np.mean(all_19_points_rmse, axis=1).reshape(1, 3)

    all_results_Transpose = np.concatenate((all_19_points_mae_Transpose, all_19_points_rmse_Transpose), axis=1) * 100
    avg_results_Transpose = np.concatenate((avg_19_points_mae_xyz, avg_19_points_rmse_xyz), axis=1) * 100

    results = np.concatenate((all_results_Transpose, avg_results_Transpose), axis=0)
    results = np.around(results, 2)
    results = results[:, [0, 3, 1, 4, 2, 5]]
    return results



def define_LSTM(input_shape, n_keypoints, n_units):
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
    model.add(LSTM(units=n_units, return_sequences=False))

    # Output layer
    model.add(Dense(n_keypoints, activation='linear'))

    # compile the model
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    
    return model


def training_loop(model, model_name, featuremap_train, labels_train, featuremap_validate, labels_validate, featuremap_test, labels_test, batch_size=128, epochs=150, n_iterations=10):
    # Initialize the list to store the result for each iteration
    paper_result_list = []
    
    # Repeat i iteration to get the average result
    for i in range(n_iterations):
        print('Iteration', i)

        # instantiate the model
        keypoint_model = model
            
        # initial maximum error 
        score_min = 10
        

        history = keypoint_model.fit(featuremap_train, labels_train,
                                    batch_size=batch_size, epochs=epochs, verbose=1, 
                                    validation_data=(featuremap_validate, labels_validate))
                                    # callbacks=[tensorboard_callback, csv_logger])
        
        print(keypoint_model.summary())

        # save and print the metrics
        score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
        print('train MAPE = ', score_train[3])

        score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
        print('test MAPE = ', score_test[3])
        
        result_test = keypoint_model.predict(featuremap_test)
       
        
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
