import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_irregular_curved_rectangle(center, width, height, corner_radius, num_points, noise_scale):
    num_side_points = num_points // 4

    # Generate points for each straight side
    top = np.linspace(-width / 2 + corner_radius, width / 2 - corner_radius, num_side_points)
    bottom = np.linspace(-width / 2 + corner_radius, width / 2 - corner_radius, num_side_points)
    left = np.linspace(-height / 2 + corner_radius, height / 2 - corner_radius, num_side_points)
    right = np.linspace(-height / 2 + corner_radius, height / 2 - corner_radius, num_side_points)

    # Generate points for each corner (quarter circles)
    corner_angles = np.linspace(0, np.pi / 2, num_side_points // 4)
    top_right_corner = np.array([corner_radius * np.cos(corner_angles), corner_radius * np.sin(corner_angles)]).T
    bottom_right_corner = np.array([corner_radius * np.cos(corner_angles - np.pi / 2), corner_radius * np.sin(corner_angles - np.pi / 2)]).T
    bottom_left_corner = np.array([corner_radius * np.cos(corner_angles - np.pi), corner_radius * np.sin(corner_angles - np.pi)]).T
    top_left_corner = np.array([corner_radius * np.cos(corner_angles - 3 * np.pi / 2), corner_radius * np.sin(corner_angles - 3 * np.pi / 2)]).T

    # Assemble the rectangle with curved corners
    x = np.concatenate([
        top,
        top_right_corner[:, 0] + width / 2 - corner_radius,
        np.full(num_side_points, width / 2),
        bottom_right_corner[:, 0] + width / 2 - corner_radius,
        bottom,
        bottom_left_corner[:, 0] - width / 2 + corner_radius,
        np.full(num_side_points, -width / 2),
        top_left_corner[:, 0] - width / 2 + corner_radius
    ])
    y = np.concatenate([
        np.full(num_side_points, height / 2),
        top_right_corner[:, 1] + height / 2 - corner_radius,
        right,
        bottom_right_corner[:, 1] - height / 2 + corner_radius,
        np.full(num_side_points, -height / 2),
        bottom_left_corner[:, 1] - height / 2 + corner_radius,
        left,
        top_left_corner[:, 1] + height / 2 - corner_radius
    ])

    # Add controlled noise to each coordinate
    x_noise = np.random.normal(scale=noise_scale, size=x.shape)
    y_noise = np.random.normal(scale=noise_scale, size=y.shape)
    x += x_noise
    y += y_noise

    x += center[0]
    y += center[1]

    return np.vstack((x, y)).T

def save_shape_to_csv(shape_generator, center, num_points, noise_scale, shape_type, index, *params):
    data = shape_generator(center, *params, num_points, noise_scale)
    df = pd.DataFrame(data, columns=['X', 'Y'])
    df['ShapeID'] = 2  # all points belong to one shape
    df['PathID'] = 0  # all points belong to one path
    df = df[['ShapeID', 'PathID', 'X', 'Y']]  # reorder columns

    filename = f"{shape_type}_irregular_{index:02d}.csv"
    df.to_csv(filename, index=False)
    return filename

# Parameters
num_files = 30
num_points_per_shape = 100  # Increase to have more points for curvature
noise_scale = 0.03  # Control the scale of noise for subtle irregularities

# Generate and save shapes
rectangle_widths = np.linspace(1, 5, num_files)
rectangle_heights = np.linspace(0.5, 2.5, num_files)
corner_radii = np.linspace(0.1, 0.5, num_files)
filenames = []

for i in range(num_files):
    filenames.append(save_shape_to_csv(generate_irregular_curved_rectangle, [0, 0], num_points_per_shape, noise_scale, 'curved_rectangle', i+1, rectangle_widths[i], rectangle_heights[i], corner_radii[i]))

filenames

# Plotting the last generated curved rectangle to verify
curved_rectangle_data = generate_irregular_curved_rectangle([0, 0], rectangle_widths[-1], rectangle_heights[-1], corner_radii[-1], num_points_per_shape, noise_scale)
plt.plot(curved_rectangle_data[:, 0], curved_rectangle_data[:, 1], 'b-')
plt.axis('equal')
plt.show()
