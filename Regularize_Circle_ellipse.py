import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

def read_polyline_csv(csv_path):
    """Read X and Y points from a CSV file while ignoring other columns."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    # Select only X and Y columns, assuming they are named 'X' and 'Y'.
    points = np.vstack((data['X'], data['Y'])).T
    # Filter out rows with NaNs or Infs in either the X or Y columns.
    return points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]

def ellipse_error(params, points):
    """Calculate the total squared error of points from the ellipse defined by params."""
    x0, y0, a, b = params
    errors = ((points[:, 0] - x0) / a) ** 2 + ((points[:, 1] - y0) / b) ** 2 - 1
    return np.sum(errors ** 2)

def regularize_to_ellipse(points):
    """Adjust points to fit them onto an ellipse."""
    x0, y0 = np.mean(points, axis=0)
    a = (np.max(points[:, 0]) - np.min(points[:, 0])) / 2
    b = (np.max(points[:, 1]) - np.min(points[:, 1])) / 2
    initial_guess = [x0, y0, a, b]
    result = minimize(ellipse_error, initial_guess, args=(points,), method='L-BFGS-B')
    return result.x

def plot_regularized_ellipse(ellipse_params):
    """Plot only the regularized ellipse with symmetry lines."""
    fig, ax = plt.subplots()
    ellipse_t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = ellipse_params[0] + ellipse_params[2] * np.cos(ellipse_t)
    ellipse_y = ellipse_params[1] + ellipse_params[3] * np.sin(ellipse_t)
    ax.plot(ellipse_x, ellipse_y, 'g--', label='Regularized Ellipse')

    # Add vertical symmetry line
    ax.axvline(x=ellipse_params[0], color='r', linestyle='--', label='Vertical Symmetry')
    # Add horizontal symmetry line
    ax.axhline(y=ellipse_params[1], color='b', linestyle='--', label='Horizontal Symmetry')

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    plt.show()

def write_regularized_ellipse_to_csv(ellipse_params, output_csv_path, shape_id=1, path_id=1):
    """Write the regularized ellipse points to a CSV file with ShapeID and PathID."""
    ellipse_t = np.linspace(0, 2 * np.pi, 100)
    ellipse_x = ellipse_params[0] + ellipse_params[2] * np.cos(ellipse_t)
    ellipse_y = ellipse_params[1] + ellipse_params[3] * np.sin(ellipse_t)
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for x, y in zip(ellipse_x, ellipse_y):
            writer.writerow([shape_id, path_id, x, y])

def check_Circle_ellipse(csv_path, output_csv_path, shape_id=4, path_id=0):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a curve.")
    else:
        ellipse_params = regularize_to_ellipse(points)
        plot_regularized_ellipse(ellipse_params)
        write_regularized_ellipse_to_csv(ellipse_params, output_csv_path, shape_id, path_id)

# Main usage
# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_Circle_ellipse('cluster_0_data.csv', 'output.csv')
