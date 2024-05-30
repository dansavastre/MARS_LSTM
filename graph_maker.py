import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# Load the accuracy data
LSTM_accuracy_data = np.load('models/Accuracy/LSTM_accuracy.npy')
MARS_accuracy_data = np.load('models/Accuracy/MARS_accuracy.npy')

# Extract the data
keypoint_labels = np.array(['Spine Base', 'Spine Mid', 'Neck', 'Head', 'Shoulder L', 'Elbow L', 'Wrist L', 'Shoulder R', 'Elbow R', 'Wrist R', 
                        'Hip L', 'Knee L', 'Ankle L', 'Foot L', 'Hip R', 'Knee R', 'Ankle R', 'Foot R', 'Spine Shoulder'])
n_keypoints = len(keypoint_labels)

# Get mae for each keypoint
LSTM_mae_x = LSTM_accuracy_data[:19, 0]
LSTM_mae_y = LSTM_accuracy_data[:19, 2]
LSTM_mae_z = LSTM_accuracy_data[:19, 4]

MARS_mae_x = MARS_accuracy_data[:19, 0]
MARS_mae_y = MARS_accuracy_data[:19, 2]
MARS_mae_z = MARS_accuracy_data[:19, 4]

# Calculate improvement
improvement_x = MARS_mae_x - LSTM_mae_x
improvement_y = MARS_mae_y - LSTM_mae_y
improvement_z = MARS_mae_z - LSTM_mae_z

lstm_mae = np.concatenate([LSTM_mae_x, LSTM_mae_y, LSTM_mae_z])
mars_mae = np.concatenate([MARS_mae_x, MARS_mae_y, MARS_mae_z])
improvement = np.concatenate([improvement_x, improvement_y, improvement_z])

print(lstm_mae.shape)
print(mars_mae.shape)
print(improvement.shape)

# Create a dataframe
df = pd.DataFrame({
    'Keypoint': np.tile(keypoint_labels, 3),
    'Axis': np.repeat(['X', 'Y', 'Z'], n_keypoints),
    'LSTM MAE': lstm_mae,
    'MARS MAE': mars_mae,
    'Improvement': improvement
})

print(df)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

bars = sns.barplot(x='Keypoint', y='LSTM MAE', hue='Axis', data=df, ax=ax)

for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    # Find out if the bar is for x, y, or z
    axis = i // n_keypoints
    if axis == 0:
        ax.bar(bar.get_x() + bar.get_width() / 2, mars_mae[i], width=bar.get_width(), bottom=height, color='blue', label='Improvement' if i == 0 else "")
    elif axis == 1:
        ax.bar(bar.get_x() + bar.get_width() / 2, mars_mae[i], width=bar.get_width(), bottom=height, color='orange', label='Improvement' if i == 0 else "")
    else:
        ax.bar(bar.get_x() + bar.get_width() / 2, improvement[i], width=bar.get_width(), bottom=height, color='green', label='Improvement' if i == 0 else "")

# Rotate the x-axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


# Add labels, title, and legend
ax.set_xlabel('Keypoint', weight='black')
ax.set_ylabel('MAE', weight='black')
ax.set_title('MAE for Each Keypoint', weight='black')
ax.legend()