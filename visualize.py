import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

connections = [
    (0, 1),  # SpineBase to SpineMid
    (1, 2),  # SpineMid to Neck
    (2, 3),  # Neck to Head
    (2, 4),  # Neck to ShoulderLeft
    (2, 7),  # Neck to ShoulderRight
    (4, 5),  # ShoulderLeft to ElbowLeft
    (5, 6),  # ElbowLeft to WristLeft
    (7, 8),  # ShoulderRight to ElbowRight
    (8, 9),  # ElbowRight to WristRight
    (0, 14),  # SpineBase to HipRight
    (14, 15),  # HipRight to KneeRight
    (15, 16),  # KneeRight to AnkleRight
    (16, 17),  # AnkleRight to FootRight
    (0, 10),  # SpineBase to HipLeft
    (10, 11),  # HipLeft to KneeLeft
    (11, 12),  # KneeLeft to AnkleLeft
    (12, 13),  # AnkleLeft to FootLeft
    (2, 18),  # Neck to SpineShoulder
]

# Define keypoint colors
keypoint_colors = [
    "blue",  # SpineBase,
    "blue",  # SpineMid,
    "blue",  # Neck,
    "red",  # Head,
    "blue",  # ShoulderLeft,
    "green",  # ElbowLeft,
    "green",  # WristLeft,
    "blue",  # ShoulderRight,
    "green",  # ElbowRight,
    "green",  # WristRight,
    "blue",  # HipLeft,
    "green",  # KneeLeft,
    "green",  # AnkleLeft,
    "green",  # FootLeft,
    "blue",  # HipRight,
    "green",  # KneeRight,
    "green",  # AnkleRight,
    "green",  # FootRight,
    "blue",  # SpineShoulder
]


def plot_skeleton(reshaped_data, ax, color_default=True):
    for connection in connections:
        x_values = [reshaped_data[0][connection[0]], reshaped_data[0][connection[1]]]
        y_values = [reshaped_data[1][connection[0]], reshaped_data[1][connection[1]]]
        z_values = [reshaped_data[2][connection[0]], reshaped_data[2][connection[1]]]

        ax.plot(x_values, z_values, y_values, color="black")

    for keypoint_index in range(len(reshaped_data[0])):
        if color_default:
            color = keypoint_colors[keypoint_index]
        else:
            color = "gray"

        ax.scatter(
            reshaped_data[0][keypoint_index],
            reshaped_data[2][keypoint_index],
            reshaped_data[1][keypoint_index],
            c=color,
            marker="o",
            s=100 if keypoint_index == 3 else 50,  # Larger size for the head
        )


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Load MARS model
model = tf.keras.models.load_model("model/MARS_LSTM_2.h5")

# Add your own path of the testing data and labels
# featuremap_test = np.load("../mmWave_MSc/dataset/formatted/mmWave/testing_mmWave.npy")
# ground_truth = np.load("../mmWave_MSc/dataset/formatted/kinect/testing_labels.npy")
featuremap_test = np.load("feature/featuremap_test.npy")
ground_truth = np.load("feature/labels_test.npy")

predictions = model.predict(featuremap_test)

for predict_num, prediction in enumerate(predictions):
    ax.clear()  # Clear the plot before each iteration

    # NOTE: MARS outputs the keypoint coords as [x1, x2, ..., xN, z1, z2, ..., zN, y1, y2, ..., yN]
    reshaped_data = prediction.reshape(3, -1)
    plot_skeleton(reshaped_data, ax)

    # GROUND TRUTH (in grey)
    reshaped_ground_truth = ground_truth[predict_num].reshape(3, -1)
    plot_skeleton(reshaped_ground_truth, ax, False)

    # Set fixed axis scales
    ax.set_xlim(-2, 2)
    ax.set_ylim(0, 4)
    ax.set_zlim(0, 3)

    plt.draw()
    plt.pause(0.1)

plt.show()  # Move plt.show() outside of the loop to display the final plot
