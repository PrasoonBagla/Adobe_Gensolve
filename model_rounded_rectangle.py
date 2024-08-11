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
import joblib
 
# Directory containing shape CSV files
data_dir = 'shapes/'
 
# Directory to save images
image_dir = 'rounded_rectangle_images/'
os.makedirs(image_dir, exist_ok=True)
 
# Load labels
labels_file_path = os.path.join(data_dir, 'labels.csv')
if not os.path.exists(labels_file_path):
    raise FileNotFoundError(f"Labels file not found: {labels_file_path}")
labels = pd.read_csv(labels_file_path)
 
# Function to load data and convert to image
def shape_to_image(file_path, img_size=64):
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
 
# Save images and labels
images = []
image_labels = []
 
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv') and file_name != 'labels.csv':
        file_path = os.path.join(data_dir, file_name)
        try:
            shape_data = pd.read_csv(file_path)
            if 'ShapeID' not in shape_data.columns:
                raise ValueError(f"Column 'ShapeID' not found in file: {file_path}")
            shape_id = shape_data['ShapeID'].iloc[0]
            label = labels[labels['ShapeID'] == shape_id]['is_rounded_rectangle'].values[0]
            img = shape_to_image(file_path)
            images.append(img)
            image_labels.append(label)
            cv2.imwrite(os.path.join(image_dir, f'{shape_id}.png'), img)
        except ValueError as e:
            print(f"Error processing file {file_path}: {e}")
 
images = np.array(images)
image_labels = np.array(image_labels)
 
# Preprocess images for CNN
if images.size == 0 or image_labels.size == 0:
    raise ValueError("No valid images or labels found. Please check your data.")
 
images = images.reshape(-1, 64, 64, 1).astype('float32') / 255.0
image_labels = to_categorical(image_labels, num_classes=2)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, image_labels, test_size=0.2, random_state=42)
 
# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
 
# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
 
# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
 
# Train the model
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
 
# Save the model
model.save('models/cnn_model_rounded_rectangle.h5')
 
def test_model(file_path, model, img_size=64):
    try:
        img = shape_to_image(file_path, img_size)
        img = img.reshape(1, img_size, img_size, 1).astype('float32') / 255.0
        prediction = model.predict(img)
        return np.argmax(prediction)
    except ValueError as e:
        print(f"Error processing file {file_path}: {e}")
        return None
 
# Load the trained model
model = tf.keras.models.load_model('models/cnn_model_rounded_rectangle.h5')
 
# Example usage
test_file_path = 'shapes/circle_irregular_02.csv'
if os.path.exists(test_file_path):
    prediction = test_model(test_file_path, model)
    if prediction is not None:
        if prediction == 1:
            print(f'The shape in {test_file_path} is predicted to be a rounded rectangle.')
        else:
            print(f'The shape in {test_file_path} is predicted to not be a rounded rectangle.')
else:
    print(f'Test file not found: {test_file_path}')