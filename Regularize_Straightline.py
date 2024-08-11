import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

def read_polyline_csv(csv_path):
    """Read X and Y points from a CSV file and handle NaNs or infs."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    points = np.vstack((data['X'], data['Y'])).T
    return points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]

def line_error(params, points):
    """Calculate the total squared error of points from the line defined by params."""
    a, b = params
    errors = points[:, 1] - (a * points[:, 0] + b)
    return np.sum(errors ** 2)

def fit_line(points):
    """Fit a straight line to the given points."""
    initial_guess = [0, 0]  # Initial guess for the slope and intercept
    result = minimize(line_error, initial_guess, args=(points,), method='L-BFGS-B')
    return result.x

def plot_line(points, line_params):
    """Plot the fitted line along with the original points and symmetry lines."""
    fig, ax = plt.subplots()
    a, b = line_params
    x_values = np.array([np.min(points[:, 0]), np.max(points[:, 0])])
    y_values = a * x_values + b
    centroid = np.mean(points, axis=0)

    # Plot the fitted line
    ax.plot(x_values, y_values, 'g--', label='Fitted Line')

    # Plot the original points
    ax.plot(points[:, 0], points[:, 1], 'ro', label='Original Points')

    # Add vertical symmetry line
    ax.axvline(x=centroid[0], color='r', linestyle='--', label='Vertical Symmetry')

    # Add horizontal symmetry line
    ax.axhline(y=centroid[1], color='b', linestyle='--', label='Horizontal Symmetry')

    ax.set_aspect('equal')
    ax.legend()
    # plt.show()

def write_line_to_csv(line_params, points, output_csv_path, shape_id=1, path_id=1):
    """Write the line points to a CSV file with ShapeID and PathID."""
    a, b = line_params
    x_values = np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), num=100)
    y_values = a * x_values + b

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for x, y in zip(x_values, y_values):
            writer.writerow([shape_id, path_id, x, y])

# Main usage
def check_line(csv_path, output_csv_path, shape_id=3, path_id=1):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a line.")
    else:
        line_params = fit_line(points)
        plot_line(points, line_params)
        write_line_to_csv(line_params, points, output_csv_path, shape_id, path_id)

# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_line('shapes/line_irregular_01.csv', 'output.csv')
