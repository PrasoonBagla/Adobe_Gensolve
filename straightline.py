import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_irregular_line(start, end, num_points, noise_scale):
    x = np.linspace(start[0], end[0], num_points)
    y = np.linspace(start[1], end[1], num_points)
    
    # Add controlled noise to each coordinate
    x_noise = np.random.normal(scale=noise_scale, size=x.shape)
    y_noise = np.random.normal(scale=noise_scale, size=y.shape)
    x += x_noise
    y += y_noise

    return np.vstack((x, y)).T

def save_shape_to_csv(shape_generator, start, end, num_points, noise_scale, shape_type, index):
    data = shape_generator(start, end, num_points, noise_scale)
    df = pd.DataFrame(data, columns=['X', 'Y'])
    df['ShapeID'] = 3  # all points belong to one shape
    df['PathID'] = 0  # all points belong to one path
    df = df[['ShapeID', 'PathID', 'X', 'Y']]  # reorder columns

    filename = f"{shape_type}_irregular_{index:02d}.csv"
    df.to_csv(filename, index=False)
    return filename

# Parameters
num_files = 30
num_points_per_line = 100
noise_scale = 0.009  # Control the scale of noise for subtle irregularities

# Generate and save shapes
start_points = np.random.rand(num_files, 2) * 10 - 5  # Random start points within a 10x10 area centered at (0, 0)
end_points = np.random.rand(num_files, 2) * 10 - 5  # Random end points within a 10x10 area centered at (0, 0)
filenames = []

for i in range(num_files):
    filenames.append(save_shape_to_csv(generate_irregular_line, start_points[i], end_points[i], num_points_per_line, noise_scale, 'line', i+1))

filenames

# Plotting the last generated line to verify
line_data = generate_irregular_line(start_points[-1], end_points[-1], num_points_per_line, noise_scale)
plt.plot(line_data[:, 0], line_data[:, 1], 'b-')
plt.axis('equal')
plt.show()
