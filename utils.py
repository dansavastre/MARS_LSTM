import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def create_sequences(data, labels, seq_length, step=1):
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

