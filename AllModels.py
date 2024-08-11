import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
 
def shape_to_image_polygon(file_path, img_size=64):
    data = pd.read_csv(file_path)
    if 'X' not in data.columns or 'Y' not in data.columns:
        raise ValueError(f"Columns 'X' and 'Y' not found in file: {file_path}")
    
    # Check if data is not empty
    if data.empty:
        raise ValueError(f"No data found in file: {file_path}")
    
    # Create a blank image
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # Normalize coordinates to fit in the image
    x = ((data['X'] - data['X'].min()) / (data['X'].max() - data['X'].min()) * (img_size - 1)).astype(int)
    y = ((data['Y'] - data['Y'].min()) / (data['Y'].max() - data['Y'].min()) * (img_size - 1)).astype(int)
    
    # Check if there are enough points to draw
    if len(x) < 2 or len(y) < 2:
        raise ValueError(f"Not enough points to draw in file: {file_path}")
    
    # Draw the shape on the image
    for i in range(len(x) - 1):
        cv2.line(img, (x[i], y[i]), (x[i+1], y[i+1]), 255, 1)
    # Connect the last point to the first
    # if len(x) > 1:
    #     cv2.line(img, (x[-1], y[-1]), (x[0], y[0]), 255, 1)
    
    return img

def test_model_polygon(file_path, model, img_size=64):
    try:
        img = shape_to_image_polygon(file_path, img_size)
        img = img.reshape(1, img_size, img_size, 1).astype('float32') / 255.0
        prediction = model.predict(img)
        return np.argmax(prediction)
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return None
 
# Load the trained model
model = tf.keras.models.load_model('models/cnn_model_polygon1.h5')
 
# Example usage
test_file_path = 'shapes/isolated.csv'
if os.path.exists(test_file_path):
    prediction = test_model_polygon(test_file_path, model)
    if prediction is not None:
        if prediction == 1:
            print(f'The shape in {test_file_path} is predicted to be a regular polygon.')
        else:
            print(f'The shape in {test_file_path} is predicted to not be a regular polygon.')
else:
    print(f'Test file not found: {test_file_path}')