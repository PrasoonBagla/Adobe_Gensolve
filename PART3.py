import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

def read_and_group_data(csv_path):
    """Reads the CSV file and groups data by 'ShapeID'."""
    data = pd.read_csv(csv_path)
    grouped = data.groupby('ShapeID')
    return grouped

def plot_and_complete_curves(grouped_data):
    """Plots each group and uses cubic spline interpolation to complete the curves."""
    plt.figure(figsize=(10, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(grouped_data)))
    
    for (name, group), color in zip(grouped_data, colors):
        x = group['X'].values
        y = group['Y'].values

        # Ensure the curve is closed for periodic boundary condition
        if not np.array_equal(group.iloc[0][['X', 'Y']], group.iloc[-1][['X', 'Y']]):
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            cs = CubicSpline(np.arange(len(x)), np.c_[x, y], bc_type='periodic')
        else:
            cs = CubicSpline(np.arange(len(x)), np.c_[x, y], bc_type='natural')
        
        t_new = np.linspace(0, len(x) - 1, 300)  # Increase the number of points for smoothness
        spline_points = cs(t_new)
        
        # Plot original points
        plt.scatter(x, y, color=color, label=f'ShapeID {name}')
        
        # Plot spline curve
        plt.plot(spline_points[:, 0], spline_points[:, 1], color=color, linestyle='-', linewidth=2)
        
    plt.title('Completed Curves by ShapeID with Cubic Spline Interpolation')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

# Path to the CSV file
csv_path = 'occlusion2.csv'  # Replace with the actual path to your CSV file

# Read and group data
grouped_data = read_and_group_data(csv_path)

# Plot and complete curves
plot_and_complete_curves(grouped_data)
