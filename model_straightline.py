import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
 
# Directory containing shape CSV files
data_dir = 'shapes/'

# Load labels
labels_file_path = os.path.join(data_dir, 'labels.csv')
if not os.path.exists(labels_file_path):
    raise FileNotFoundError(f"Labels file not found: {labels_file_path}")
labels = pd.read_csv(labels_file_path)
 
# Function to calculate features from shape data
def calculate_features(file_path):
    data = pd.read_csv(file_path)
    if 'X' not in data.columns or 'Y' not in data.columns:
        raise ValueError(f"Columns 'X' and 'Y' not found in file: {file_path}")
    
    # Check if data is not empty
    if data.empty:
        raise ValueError(f"No data found in file: {file_path}")
    
    # Get the coordinates
    x = data['X'].values
    y = data['Y'].values
    
    # Calculate angles between consecutive points
    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        angles.append(angle)
    
    if len(angles) == 0:
        angle_variance = 0.0
    else:
        angle_variance = np.var(angles)
 
    # Calculate total length of the shape
    total_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    return [angle_variance, total_length]
 
# Collect all data
features = []
labels_list = []
 
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv') and file_name != 'labels.csv':
        file_path = os.path.join(data_dir, file_name)
        try:
            shape_id = pd.read_csv(file_path)['ShapeID'].iloc[0]
            label = labels[labels['ShapeID'] == shape_id]['is_straight_line'].values[0]
            feature = calculate_features(file_path)
            features.append(feature)
            labels_list.append(label)
        except ValueError as e:
            print(f"Error processing file {file_path}: {e}")
 
features = np.array(features)
labels_list = np.array(labels_list)
 
# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
# Save the scaler
scaler_path = 'models/line_scaler.pkl'
joblib.dump(scaler, scaler_path)
 
# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_list, test_size=0.2, random_state=42)
 
# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
 
# Save the model
model_path = 'models/line_model.pkl'
joblib.dump(clf, model_path)
 
# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
 
# Function to test the model on an individual file
def test_model(file_path):
    feature = calculate_features(file_path)
    feature = scaler.transform([feature])
    prediction = clf.predict(feature)
    return prediction[0]
 
# Example usage
test_file_path = 'cluster_0_data.csv'
if os.path.exists(test_file_path):
    prediction = test_model(test_file_path)
    if prediction == 1:
        print(f'The shape in {test_file_path} is predicted to be a straight line.')
    else:
        print(f'The shape in {test_file_path} is predicted to not be a straight line.')
else:
    print(f'Test file not found: {test_file_path}')