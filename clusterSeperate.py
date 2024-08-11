import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def classify_shape(hull, points):
    hull_path = points[hull.vertices]
    x_min, x_max = hull_path[:, 0].min(), hull_path[:, 0].max()
    y_min, y_max = hull_path[:, 1].min(), hull_path[:, 1].max()
    aspect_ratio = (x_max - x_min) / (y_max - y_min)

    edges = np.sqrt((np.diff(hull_path[:, 0], append=hull_path[0, 0]))**2 +
                    (np.diff(hull_path[:, 1], append=hull_path[0, 1]))**2)
    cos_angles = []
    for i in range(len(hull.vertices)):
        p1 = hull_path[i - 1]
        p2 = hull_path[i]
        p3 = hull_path[(i + 1) % len(hull.vertices)]
        vec1 = p1 - p2
        vec2 = p3 - p2
        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        cos_angles.append(cosine_angle)
    
    if np.std(edges) < 0.1 and np.std(cos_angles) < 0.1:
        if len(edges) > 4:
            return "Regular Polygon"
        elif len(edges) == 4 and aspect_ratio > 0.9 and aspect_ratio < 1.1:
            return "Rectangle"
    if aspect_ratio > 0.9 and aspect_ratio < 1.1:
        return "Circle or Ellipse"

    return "Irregular"

# Load data
def ClusterSeperate():
    data = pd.read_csv('isolated.csv')
    # Normalize the coordinates
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['X', 'Y']])

    # Apply DBSCAN to cluster points - Tune eps and min_samples
    dbscan = DBSCAN(eps=0.1, min_samples=10)
    clusters = dbscan.fit_predict(scaled_data)

    # Add cluster labels back to the dataframe
    data['Cluster'] = clusters

    # Filter out noise points (DBSCAN labels noise as -1)
    filtered_data = data[data['Cluster'] != -1]

    # Plotting setup
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(filtered_data['Cluster'].unique())))

    for cluster_id, color in zip(filtered_data['Cluster'].unique(), colors):
        cluster_points = filtered_data[filtered_data['Cluster'] == cluster_id]
        points = cluster_points[['X', 'Y']].values
        if len(points) > 2:
            hull = ConvexHull(points)
            shape_type = classify_shape(hull, points)
            poly = Polygon(points[hull.vertices], closed=True, fill=None, edgecolor=color, label=f'Cluster {cluster_id}: {shape_type}')
            ax.add_patch(poly)
            ax.scatter(points[:, 0], points[:, 1], color=color, s=10)
            centroid = points.mean(axis=0)
            ax.text(centroid[0], centroid[1], f'Cluster {cluster_id}: {shape_type}', color='black', fontsize=12)
            
            # Save cluster data to CSV
            cluster_filename = f'Clusters/cluster_{cluster_id}_data.csv'
            cluster_points.to_csv(cluster_filename, index=False)
            print(f'Saved Cluster {cluster_id} data to {cluster_filename}')

    ax.set_title('Cluster Visualization with Convex Hulls and Shape Classification')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend()
    # plt.show()

ClusterSeperate()