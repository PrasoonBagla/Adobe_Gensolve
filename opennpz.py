import numpy as np
import pandas as pd

try:
    # Load the NPZ file
    data = np.load('test.npz')

    # Extract arrays
    x = data['x']
    y = data['y']

    # Create a DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y
    })

    # Save the DataFrame to a CSV file
    df.to_csv('outputnpz.csv', index=False)

except OSError as e:
    print("OS error occurred:", e)

except Exception as e:
    print("An error occurred:", e)
