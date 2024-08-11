import numpy as np
import matplotlib.pyplot as plt

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',') 
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:] 
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY) 
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']  # Define a list of colours
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs): 
        c = colours[i % len(colours)]  # Use modulo to cycle through the colours list
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Assuming the CSV path is correctly set
path = read_csv("Output/combined_output.csv")  # Uncomment this line if the CSV file path is correct
plot(path)
