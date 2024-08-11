import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import csv

def read_polyline_csv(csv_path):
    """Read X and Y points from a CSV file while ignoring other columns."""
    data = np.genfromtxt(csv_path, delimiter=',', names=True)
    points = np.vstack((data['X'], data['Y'])).T
    return points[~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)]

def calculate_centroid(points):
    """Calculate the geometric centroid of a set of points."""
    return np.mean(points, axis=0)

def calculate_polar_coordinates(points, center):
    """Calculate the polar coordinates (radius, angle) for each point from the center."""
    relative_points = points - center
    angles = np.arctan2(relative_points[:, 1], relative_points[:, 0])
    radii = np.sqrt(np.sum(relative_points**2, axis=1))
    return radii, angles, relative_points

def find_star_tips_and_troughs(points, radii, angles):
    """Identify peaks and troughs in the radii based on angular sorting with wrap-around consideration."""
    sorted_indices = np.argsort(angles)
    sorted_radii = radii[sorted_indices]
    sorted_points = points[sorted_indices]
    sorted_angles = angles[sorted_indices]

    # Extend angles and radii to handle wrap-around
    extended_angles = np.concatenate([sorted_angles - 2 * np.pi, sorted_angles, sorted_angles + 2 * np.pi])
    extended_radii = np.concatenate([sorted_radii, sorted_radii, sorted_radii])
    extended_points = np.concatenate([sorted_points, sorted_points, sorted_points])

    # Find peaks in the extended array
    peaks, _ = find_peaks(extended_radii, prominence=0.05)  # Adjust prominence as needed
    peak_indices = peaks[(peaks >= len(sorted_points)) & (peaks < 2 * len(sorted_points))] - len(sorted_points)
    peak_points = sorted_points[peak_indices]

    # Find troughs between peaks
    trough_points = []
    extended_peaks = np.r_[peak_indices, peak_indices[0] + len(sorted_points)]
    for i in range(len(peak_indices)):
        segment_start = extended_peaks[i]
        segment_end = extended_peaks[i+1] if i < len(extended_peaks) - 1 else extended_peaks[0] + len(sorted_points)
        segment = sorted_radii[segment_start:segment_end]
        if len(segment) > 1:
            trough_index = np.argmin(segment) + segment_start
            trough_points.append(sorted_points[trough_index % len(sorted_points)])
    trough_points = np.array(trough_points)

    return peak_points, trough_points

def plot_star_shape(center, peak_points, trough_points):
    """Plot the star outline formed by connecting peaks and troughs with symmetry lines."""
    fig, ax = plt.subplots()
    ax.plot(center[0], center[1], 'bo', label='Center')  # Mark the center

    # Connect peaks and troughs to form the star
    star_points = np.vstack([val for pair in zip(peak_points, trough_points) for val in pair])
    star_points = np.vstack([star_points, star_points[0]])  # Close the loop
    ax.plot(star_points[:, 0], star_points[:, 1], 'g-', label='Star Outline')

    # Add vertical symmetry line
    ax.axvline(x=center[0], color='r', linestyle='--', label='Vertical Symmetry')
    # Add horizontal symmetry line
    ax.axhline(y=center[1], color='b', linestyle='--', label='Horizontal Symmetry')

    ax.set_aspect('equal')
    plt.legend()
    # plt.show()

def write_star_shape_to_csv(center, peak_points, trough_points, output_csv_path, shape_id=1, path_id=1):
    """Write the star shape points to a CSV file with ShapeID and PathID."""
    star_points = np.vstack([val for pair in zip(peak_points, trough_points) for val in pair])
    star_points = np.vstack([star_points, star_points[0]])  # Close the loop

    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ShapeID', 'PathID', 'X', 'Y'])
        for point in star_points:
            writer.writerow([shape_id, path_id, point[0], point[1]])

# Main usage
def check_star(csv_path, output_csv_path, shape_id=5, path_id=0):
    points = read_polyline_csv(csv_path)
    if points.size == 0:
        print("No valid data points to fit a curve.")
    else:
        center = calculate_centroid(points)
        radii, angles, _ = calculate_polar_coordinates(points, center)
        peak_points, trough_points = find_star_tips_and_troughs(points, radii, angles)
        plot_star_shape(center, peak_points, trough_points)
        write_star_shape_to_csv(center, peak_points, trough_points, output_csv_path, shape_id, path_id)

# Replace 'input.csv' with your actual input file path and 'output.csv' with your desired output file path.
# check_star('input.csv', 'output.csv')
