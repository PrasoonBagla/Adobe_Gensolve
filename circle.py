import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_circle(radius, center, num_points=100, shape_id=0, path_id=0):
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    data = pd.DataFrame({'ShapeID': shape_id, 'PathID': path_id, 'X': x, 'Y': y})
    return data

def generate_square(size, center, num_points=100, shape_id=0, path_id=0):
    x = np.array([])
    y = np.array([])
    steps = num_points // 4
    # Top edge
    x = np.append(x, np.linspace(center[0] - size / 2, center[0] + size / 2, steps))
    y = np.append(y, np.full(steps, center[1] + size / 2))
    # Right edge
    x = np.append(x, np.full(steps, center[0] + size / 2))
    y = np.append(y, np.linspace(center[1] + size / 2, center[1] - size / 2, steps))
    # Bottom edge
    x = np.append(x, np.linspace(center[0] + size / 2, center[0] - size / 2, steps))
    y = np.append(y, np.full(steps, center[1] - size / 2))
    # Left edge
    x = np.append(x, np.full(steps, center[0] - size / 2))
    y = np.append(y, np.linspace(center[1] - size / 2, center[1] + size / 2, steps))

    data = pd.DataFrame({'ShapeID': shape_id, 'PathID': path_id, 'X': x, 'Y': y})
    return data

# Directory to save shapes CSV files
shapes_dir = 'shape_data'
os.makedirs(shapes_dir, exist_ok=True)

# Generate and save shapes
shapes = ['circle', 'square']
for i, shape in enumerate(shapes):
    if shape == 'circle':
        data = generate_circle(radius=50, center=(50, 50), shape_id=i)
    elif shape == 'square':
        data = generate_square(size=100, center=(50, 50), shape_id=i)
    
    filename = os.path.join(shapes_dir, f'{shape}_{i}.csv')
    data.to_csv(filename, index=False)
    plt.plot(data['X'], data['Y'], label=f'{shape} {i}')

plt.legend()
plt.axis('equal')
plt.show()
