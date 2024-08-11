import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import tensorflow as tf
from Regularize_Circle_ellipse import check_Circle_ellipse
from Regularize_Rounded_Rectangle import check_Rounded_Rectangle
from Regularize_Rectangle import check_Rectangle
from Regularize_star import check_star
from clusterSeperate import ClusterSeperate
from Regularize_Polygon import check_Polygon
from Regularize_Straightline import check_line
from sklearn.decomposition import PCA
import joblib

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',') 
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:] 
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY) 
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs): 
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    # plt.show()

def shape_to_image(file_path, img_size=64):
    data = pd.read_csv(file_path)
    if 'X' not in data.columns or 'Y' not in data.columns:
        raise ValueError(f"Columns 'X' and 'Y' not found in file: {file_path}")
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    x = ((data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min()) * (img_size - 1)).astype(int)
    y = ((data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min()) * (img_size - 1)).astype(int)
    if len(x) < 2 or len(y) < 2:
        raise ValueError(f"Not enough points to draw in file: {file_path}")
    for i in range(len(x) - 1):
        cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), 255, 1)
    return img

def calculate_features_straight_line(file_path):
    data = pd.read_csv(file_path)
    x = data['X'].values
    y = data['Y'].values
    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i - 1], y[i - 1]] - np.array([x[i], y[i]]))
        v2 = np.array([x[i + 1], y[i + 1]] - np.array([x[i], y[i]]))
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angles.append(angle)
    angle_variance = np.var(angles) if angles else 0.0
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    return [angle_variance, total_length]

def test_model(file_path, model, img_size=64):
    img = shape_to_image(file_path, img_size)
    img = img.reshape(1, img_size, img_size, 1).astype('float32') / 255.0
    prediction = model.predict(img)
    return np.argmax(prediction)

ClusterSeperate()  # Assuming this function is defined elsewhere and necessary here
directory_path = 'Clusters'
model_paths = {
    "circle": ("models/cnn_model_circle.h5", check_Circle_ellipse),
    "rounded_rectangle": ("models/cnn_model_rounded_rectangle.h5", check_Rounded_Rectangle),
    "rectangle": ("models/cnn_model_rectangle.h5", check_Rectangle),
    "star": ("models/cnn_model_star.h5", check_star),
    "polygon": ("models/cnn_model_polygon1.h5", check_Polygon)
}
i = 0

for filename in os.listdir(directory_path):
    i = i+1
    test_file_path = os.path.join(directory_path, filename)
    paths_XYs = read_csv(test_file_path)
    # plot(paths_XYs)

    for shape, (model_path, regularize_function) in model_paths.items():
        model = tf.keras.models.load_model(model_path)
        prediction = test_model(test_file_path, model)
        if prediction == 1:
            print(f"The shape in {test_file_path} is predicted to be a {shape}.")
            regularize_function(test_file_path, f"Output/{i}.csv")
            break

    else:
        scaler = joblib.load('models/line_scaler.pkl')
        model = joblib.load('models/line_model.pkl')
        features = calculate_features_straight_line(test_file_path)
        features = scaler.transform([features])
        prediction = model.predict(features)
        if prediction[0] == 1:
            print(f"The shape in {test_file_path} is predicted to be a straight line.")
            check_line(test_file_path, "Output/{i}.csv")

output_directory = 'Output'
output_filename = 'combined_output.csv'

data_frames = []
first_file = True

# Loop over each file in the output directory
for filename in os.listdir(output_directory):
    file_path = os.path.join(output_directory, filename)
    if file_path.endswith('.csv'):
        # Read the CSV file
        if first_file:
            # Keep the header for the first file
            data_frame = pd.read_csv(file_path)
            first_file = False
        else:
            # Skip the header for subsequent files
            data_frame = pd.read_csv(file_path, header=0)
        
        # Store the DataFrame in the list
        data_frames.append(data_frame)

# Concatenate all data frames into a single data frame
combined_data_frame = pd.concat(data_frames, ignore_index=True)

# Save the combined data frame to a new CSV file
# combined_data_frame.to_csv(os.path.join(output_directory, output_filename), index=False)
combined_data_frame.to_csv('combined_output.csv', index=False)
print(f"All files have been combined into {os.path.join(output_directory, output_filename)}")

def delete_files_in_directory(directory):
    # List all files and directories in the given directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # This removes files and links
            elif os.path.isdir(file_path):
                # If you want to also remove directories, uncomment the following line
                # shutil.rmtree(file_path)
                print(f"Skipping directory: {file_path}")
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
 
# Specify the directories
directories = ['Output', 'Clusters']
 
# Apply deletion function to each directory
for directory in directories:
    delete_files_in_directory(directory)
    print(f"All files in the {directory} directory have been deleted.")