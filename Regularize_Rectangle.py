import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import csv

def read_polyline_csv(csv_path):
    """Read points from a CSV file and handle NaNs or infs."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    points = np.vstack((data['X'], data['Y'])).T
    filtered_points = points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]
    print("Filtered points:", filtered_points)  # Diagnostic print
    return filtered_points

def rectangle_error(params, points):
    cx, cy, width, height, angle = params
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    transformed_points = np.dot(points - np.array([cx, cy]), rotation_matrix.T)
    dx = np.maximum(0, np.abs(transformed_points[:, 0]) - width / 2)
    dy = np.maximum(0, np.abs(transformed_points[:, 1]) - height / 2)
    return np.sum(dx**2 + dy**2)

def fit_rectangle(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    cx, cy = np.mean(hull_points, axis=0)
    width, height = np.ptp(hull_points, axis=0)
    initial_guess = [cx, cy, width, height, 0]
    result = minimize(rectangle_error, initial_guess, args=(points,), method='L-BFGS-B')
    print("Optimization result:", result)  # Diagnostic print
    return result.x

def plot_shapes(points, rect_params):
    fig, ax = plt.subplots()
    cx, cy, width, height, angle = rect_params
    print("Rectangle params:", rect_params)  # Diagnostic print
    
    # Plot original points
    ax.plot(points[:, 0], points[:, 1], 'ro', label='Original Points')  # Added to plot original points
    
    # Plot fitted rectangle
    rect = plt.Rectangle((cx - width/2, cy - height/2), width, height, angle=angle * 180 / np.pi,
                         edgecolor='blue', facecolor='none', lw=2)
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle * 180 / np.pi) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    
    # Add vertical symmetry line
    ax.axvline(x=cx, color='r', linestyle='--', label='Vertical Symmetry')
    # Add horizontal symmetry line
    ax.axhline(y=cy, color='b', linestyle='--', label='Horizontal Symmetry')

    ax.set_xlim([cx - width, cx + width])
    ax.set_ylim([cy - height, cy + height])
    ax.set_aspect('equal')
    ax.legend()
    # plt.show()

def write_rectangle_to_csv(rect_params, output_csv_path, shape_id=1, path_id=1):
    """Write the rectangle points to a CSV file with ShapeID and PathID."""
    cx, cy, width, height, angle = rect_params
    corners = np.array([
        [cx - width / 2, cy - height / 2],
        [cx + width / 2, cy - height / 2],
        [cx + width / 2, cy + height / 2],
        [cx - width / 2, cy + height / 2],
        [cx - width / 2, cy - height / 2]  # Appended first point to make a closed shape
    ])

    # Apply rotation to the corners
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_corners = np.dot(corners - np.array([cx, cy]), rotation_matrix.T) + np.array([cx, cy])

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for corner in rotated_corners:
            writer.writerow([shape_id, path_id, corner[0], corner[1]])

# Main usage
def check_Rectangle(csv_path, output_csv_path, shape_id=1, path_id=0):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a shape.")
    else:
        rect_params = fit_rectangle(points)
        plot_shapes(points, rect_params)
        write_rectangle_to_csv(rect_params, output_csv_path, shape_id, path_id)

# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_Rectangle('input.csv', 'output.csv')
