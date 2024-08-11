import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import csv

def read_polyline_csv(csv_path):
    """Read points from a CSV file and handle NaNs or infs."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    points = np.vstack((data['X'], data['Y'])).T
    return points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]

def regular_polygon_error(params, points, num_sides):
    """Calculate the total squared error of points from the regular polygon defined by params."""
    x0, y0, r = params
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    polygon_x = x0 + r * np.cos(angles)
    polygon_y = y0 + r * np.sin(angles)
    polygon_points = np.vstack((polygon_x, polygon_y)).T
    errors = np.sum(np.min(np.sum((points[:, None, :] - polygon_points[None, :, :]) ** 2, axis=2), axis=1))
    return errors

def regularize_to_polygon(points, num_sides):
    """Adjust points to fit them onto a regular polygon."""
    x0, y0 = np.mean(points, axis=0)
    r = np.mean(np.sqrt((points[:, 0] - x0) ** 2 + (points[:, 1] - y0) ** 2))
    initial_guess = [x0, y0, r]
    result = minimize(regular_polygon_error, initial_guess, args=(points, num_sides), method='L-BFGS-B')
    return result.x

def find_best_polygon(points, min_sides=3, max_sides=10):
    """Determine the best number of sides by minimizing the error."""
    best_error = float('inf')
    best_params = None
    best_sides = 0
    for num_sides in range(min_sides, max_sides + 1):
        params = regularize_to_polygon(points, num_sides)
        error = regular_polygon_error(params, points, num_sides)
        if error < best_error:
            best_error = error
            best_params = params
            best_sides = num_sides
    return best_params, best_sides

def plot_regular_polygon(points, polygon_params):
    """Plot the regular polygon along with the original points and symmetry lines."""
    fig, ax = plt.subplots()
    x0, y0, r, num_sides = *polygon_params[0], polygon_params[1]
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    polygon_x = x0 + r * np.cos(angles)
    polygon_y = y0 + r * np.sin(angles)
    polygon_x = np.append(polygon_x, polygon_x[0])  # Close the polygon
    polygon_y = np.append(polygon_y, polygon_y[0])  # Close the polygon
    ax.plot(polygon_x, polygon_y, 'g--', label=f'Regular Polygon with {num_sides} sides')
    ax.plot(points[:, 0], points[:, 1], 'ro', label='Original Points')  # Plotting the original points
    
    # Add vertical symmetry line
    ax.axvline(x=x0, color='r', linestyle='--', label='Vertical Symmetry')
    # Add horizontal symmetry line
    ax.axhline(y=y0, color='b', linestyle='--', label='Horizontal Symmetry')

    ax.set_aspect('equal', adjustable='datalim')
    ax.legend()
    # plt.show()

def write_polygon_to_csv(polygon_params, output_csv_path, shape_id=1, path_id=1):
    """Write the polygon points to a CSV file with ShapeID and PathID."""
    x0, y0, r, num_sides = *polygon_params[0], polygon_params[1]
    angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
    polygon_x = x0 + r * np.cos(angles)
    polygon_y = y0 + r * np.sin(angles)
    polygon_points = np.vstack((polygon_x, polygon_y)).T

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for point in polygon_points:
            writer.writerow([shape_id, path_id, point[0], point[1]])

# Main usage
def check_Polygon(csv_path, output_csv_path, shape_id=4, path_id=0):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a curve.")
    else:
        polygon_params = find_best_polygon(points)
        plot_regular_polygon(points, polygon_params)
        write_polygon_to_csv(polygon_params, output_csv_path, shape_id, path_id)

# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_Polygon('input.csv', 'output.csv')
