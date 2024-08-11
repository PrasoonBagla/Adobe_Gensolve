import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

# Plot the figure from the CSV data
def plot_figure(data):
    unique_shapes = data['ShapeID'].unique()
    plt.figure(figsize=(10, 6))
    for shape_id in unique_shapes:
        shape_data = data[data['ShapeID'] == shape_id]
        plt.plot(shape_data['X'], shape_data['Y'], marker='o', linestyle='-', label=f'Shape {shape_id}')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Calculate the centroid
def calculate_centroid(data):
    x_centroid = data['X'].mean()
    y_centroid = data['Y'].mean()
    return x_centroid, y_centroid

# Check for symmetry around a vertical line through the centroid
def check_vertical_symmetry(data, x_centroid):
    if 'X' not in data.columns:
        print("Error: 'X' column not found in the data")
        return False
    if 'Y' not in data.columns:
        print("Error: 'Y' column not found in the data")
        return False

    # Reflect all X points across the centroid line
    data['X_reflected'] = 2 * x_centroid - data['X']
    reflected_data = data[['X_reflected', 'Y']].rename(columns={'X_reflected': 'X'})
    original_data = data[['X', 'Y']]
    
    # Check if every reflected point has a corresponding original point
    merge_data = pd.merge(reflected_data, original_data, on=['X', 'Y'], how='inner')
    return merge_data.shape[0] == data.shape[0]

# Example usage:
filepath = 'isolated_sol.csv'
data = load_data(filepath)
print("Data Columns:", data.columns)  # Debugging line to check column names
plot_figure(data)

# Finding lines of symmetry for each shape
unique_shapes = data['ShapeID'].unique()
for shape_id in unique_shapes:
    shape_data = data[data['ShapeID'] == shape_id]
    x_centroid, y_centroid = calculate_centroid(shape_data)
    if check_vertical_symmetry(shape_data, x_centroid):
        print(f"Shape {shape_id} is symmetric about x = {x_centroid}")
    else:
        print(f"Shape {shape_id} is not symmetric about x = {x_centroid}")
