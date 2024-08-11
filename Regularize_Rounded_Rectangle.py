import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from matplotlib.patches import FancyBboxPatch
import csv

def read_polyline_csv(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    points = np.vstack((data['X'], data['Y'])).T
    filtered_points = points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]
    return filtered_points

def rectangle_error(params, points):
    cx, cy, width, height, angle, radius = params
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    transformed_points = np.dot(points - np.array([cx, cy]), rotation_matrix.T)
    dx = np.maximum(0, np.abs(transformed_points[:, 0]) - (width / 2 - radius))
    dy = np.maximum(0, np.abs(transformed_points[:, 1]) - (height / 2 - radius))
    return np.sum(dx**2 + dy**2)

def fit_rectangle(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    cx, cy = np.mean(hull_points, axis=0)
    width, height = np.ptp(hull_points, axis=0)
    radius = min(width, height) * 0.1  # Starting guess: 10% of smaller dimension
    initial_guess = [cx, cy, width, height, 0, radius]
    result = minimize(rectangle_error, initial_guess, args=(points,), method='L-BFGS-B')
    return result.x

def plot_shapes(points, rect_params):
    fig, ax = plt.subplots()
    cx, cy, width, height, angle, radius = rect_params

    # Create a FancyBboxPatch for a rounded rectangle
    box = FancyBboxPatch((cx - width / 2, cy - height / 2), width, height,
                         boxstyle="round,pad=0,rounding_size={}".format(radius),
                         ec="blue", fc="none", lw=2)

    # Transform for rotation around the center
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle * 180 / np.pi) + ax.transData
    box.set_transform(t)
    ax.add_patch(box)

    # Plot original points
    # ax.plot(points[:, 0], points[:, 1], 'ro', label='Original Points')

    # Add vertical symmetry line
    ax.axvline(x=cx, color='r', linestyle='--', label='Vertical Symmetry')
    # Add horizontal symmetry line
    ax.axhline(y=cy, color='b', linestyle='--', label='Horizontal Symmetry')

    # Set plot limits and properties
    ax.set_xlim([cx - width, cx + width])
    ax.set_ylim([cy - height, cy + height])
    ax.set_aspect('equal')
    ax.legend()
    # plt.show()

def write_rounded_rectangle_to_csv(rect_params, output_csv_path, shape_id=1, path_id=1):
    """Write the rounded rectangle points to a CSV file with ShapeID and PathID."""
    cx, cy, width, height, angle, radius = rect_params
    corners = np.array([
        [-width / 2 + radius, -height / 2 + radius],
        [width / 2 - radius, -height / 2 + radius],
        [width / 2 - radius, height / 2 - radius],
        [-width / 2 + radius, height / 2 - radius]
    ])

    # Calculate the points for the rounded rectangle
    points = []
    for corner in corners:
        arc = np.linspace(0, np.pi / 2, 25)
        for a in arc:
            points.append([
                cx + corner[0] + radius * np.cos(a),
                cy + corner[1] + radius * np.sin(a)
            ])
        corner = np.dot(corner, np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]))
        points.append([cx + corner[0], cy + corner[1]])

    points = np.array(points)

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for point in points:
            writer.writerow([shape_id, path_id, point[0], point[1]])

# Main usage
def check_Rounded_Rectangle(csv_path, output_csv_path, shape_id=2, path_id=0):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a shape.")
    else:
        rect_params = fit_rectangle(points)
        plot_shapes(points, rect_params)
        write_rounded_rectangle_to_csv(rect_params, output_csv_path, shape_id, path_id)

# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_Rounded_Rectangle('input.csv', 'output.csv')
