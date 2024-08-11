import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_star(center=(0, 0), num_arms=8, arm_length=100, noise_level=5):
    """Generate a star shape with given parameters and noise."""
    angles = np.linspace(0, 2 * np.pi, num_arms * 2, endpoint=False)
    lengths = np.array([arm_length if i % 2 == 0 else arm_length * 0.4 for i in range(len(angles))])
    
    # Adding noise to mimic hand-drawn variability
    noisy_lengths = lengths + np.random.normal(0, noise_level, lengths.shape)
    
    xs = center[0] + noisy_lengths * np.cos(angles)
    ys = center[1] + noisy_lengths * np.sin(angles)
    
    return xs, ys

# Generate star points
xs, ys = generate_star(center=(0, 0), num_arms=8, arm_length=100, noise_level=10)

# Save to CSV
df = pd.DataFrame({'x': xs, 'y': ys})
csv_path = 'simulated_star.csv'
df.to_csv(csv_path, index=False)

# Plotting for visualization
plt.figure(figsize=(6, 6))
plt.plot(xs, ys, 'ro-')  # Red dots connected by lines
plt.title("Simulated Hand-Drawn Star")
plt.axis('equal')  # Equal scaling for x and y axes
plt.grid(True)
plt.show()
