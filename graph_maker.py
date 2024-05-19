import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# Load the accuracy data
accuracy_data = np.load('model/Accuracy/MARS_accuracy.npy')

# Extract the data
joint_labels = np.array(['Spine Base', 'Spine Mid', 'Neck', 'Head', 'Shoulder L', 'Elbow L', 'Wrist L', 'Shoulder R', 'Elbow R', 'Wrist R', 
                        'Hip L', 'Knee L', 'Ankle L', 'Foot L', 'Hip R', 'Knee R', 'Ankle R', 'Foot R', 'Spine Shoulder'])
joints = np.arange(1, 20)
x_mae = accuracy_data[:-1, 0]
x_rmse = accuracy_data[:-1, 1]
y_mae = accuracy_data[:-1, 2]
y_rmse = accuracy_data[:-1, 3]
z_mae = accuracy_data[:-1, 4]
z_rmse = accuracy_data[:-1, 5]

# Create a DataFrame for easier plotting
data = {
    'Joint': np.tile(joint_labels, 3),
    'MAE': np.concatenate([x_mae, y_mae, z_mae]),
    'RMSE': np.concatenate([x_rmse, y_rmse, z_rmse]),
    'Axis': ['X'] * len(joints) + ['Y'] * len(joints) + ['Z'] * len(joints)
}

df = pd.DataFrame(data)

# Plot settings
mpl.rcParams['font.size'] = 20

# Plot MAE
plt.figure(figsize=(16, 9))
sns.barplot(x='Joint', y='MAE', hue='Axis', data=df)
plt.title('Mean Absolute Error (MAE) for Each Joint', weight='black')
plt.xlabel('Joint', weight='black')
plt.ylabel('MAE (cm)', weight='black')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot RMSE
plt.figure(figsize=(16, 9))
sns.barplot(x='Joint', y='RMSE', hue='Axis', data=df)
plt.title('Root Mean Squared Error (RMSE) for Each Joint', weight='black')
plt.xlabel('Joint', weight='black')
plt.ylabel('RMSE (cm)', weight='black')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()