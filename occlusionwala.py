import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

def fit_and_save_curve(input_csv, output_csv, degree=3):
    """
    Fits a polynomial curve to the data in the input CSV file and writes the fitted curve to an output CSV file.
    
    Parameters:
        input_csv (str): Path to the input CSV file containing the original data.
        output_csv (str): Path to the output CSV file where the fitted curve will be saved.
        degree (int): Degree of the polynomial to fit the data.
    """
    # Load data
    data = pd.read_csv(input_csv)
    x = data.iloc[:, 2].dropna()  # Assuming x-coordinates are in the third column
    y = data.iloc[:, 3].dropna()  # Assuming y-coordinates are in the fourth column

    # Fit a polynomial of specified degree
    poly = Polynomial.fit(x, y, degree)

    # Generate x values for the output curve, covering the same range
    x_new = np.linspace(x.min(), x.max(), 300)
    y_new = poly(x_new)

    # Save the new data to a CSV file
    output_data = pd.DataFrame({
        'X': x_new,
        'Y': y_new
    })
    output_data.to_csv(output_csv, index=False)

# Example usage
if __name__ == "__main__":
    input_csv_path = 'occlusion2.csv'  # Replace with the path to your input file
    output_csv_path = 'path_to_output.csv'  # Replace with the path where you want the output saved
    fit_and_save_curve(input_csv_path, output_csv_path)
