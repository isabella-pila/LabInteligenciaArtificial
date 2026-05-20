import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('treinamento.csv')
# Select patterns with presence of radiation (d = 1)
df_pos = df[df['d'] == 1].copy()
X_pos = df_pos[['x1', 'x2']].values

print(f"Total positive samples (d=1): {len(X_pos)}")
print("Positive samples:")
for i, row in enumerate(X_pos):
    print(f"Sample {i+1} (Original index in training set: {df_pos.index[i]+1}): {row}")

# Let's run a custom K-means with initialization on the first two d=1 samples
# Sample 3 in training is index 2 (0-based) -> [0.1157, 0.3676]
# Sample 4 in training is index 3 (0-based) -> [0.5147, 0.0167]
c1 = X_pos[0] # corresponding to training sample 3
c2 = X_pos[1] # corresponding to training sample 4

print("\n--- Method 1: Initialize with first two positive samples ---")
print(f"Initial c1: {c1}")
print(f"Initial c2: {c2}")

def run_kmeans(X, c1, c2, max_iters=100):
    centers = np.array([c1, c2])
    for iteration in range(max_iters):
        # Assign clusters
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Compute new centers
        new_centers = np.zeros_like(centers)
        for j in range(2):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)
            else:
                new_centers[j] = centers[j] # keep old if empty
                
        # Check convergence
        if np.allclose(centers, new_centers):
            print(f"Converged in {iteration+1} iterations.")
            break
        centers = new_centers
    
    # Calculate variances
    variances = []
    std_devs = []
    for j in range(2):
        cluster_points = X[labels == j]
        # Variance of a cluster j: mean squared distance to center
        if len(cluster_points) > 0:
            var_j = np.mean(np.sum((cluster_points - centers[j])**2, axis=1))
        else:
            var_j = 0.0
        variances.append(var_j)
        std_devs.append(np.sqrt(var_j))
        
    return centers, labels, variances, std_devs

centers, labels, variances, std_devs = run_kmeans(X_pos, c1, c2)
print("Final centers:")
print(f"Cluster 1 center: {centers[0]}, Variance: {variances[0]:.6f}, Std Dev: {std_devs[0]:.6f}, Points: {np.sum(labels == 0)}")
print(f"Cluster 2 center: {centers[1]}, Variance: {variances[1]:.6f}, Std Dev: {std_devs[1]:.6f}, Points: {np.sum(labels == 1)}")

# Print which samples are in which cluster
for j in range(2):
    print(f"\nCluster {j+1} members (Original indices):")
    members = df_pos[labels == j]
    for idx, row in members.iterrows():
        print(f"  Sample {idx+1}: [{row['x1']}, {row['x2']}]")

# Try scikit-learn's KMeans if installed to see if it gives the same or different
try:
    from sklearn.cluster import KMeans
    print("\n--- Method 2: scikit-learn KMeans ---")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_pos)
    sk_centers = kmeans.cluster_centers_
    sk_labels = kmeans.labels_
    
    # Calculate variances for sklearn
    sk_variances = []
    sk_std_devs = []
    for j in range(2):
        cluster_points = X_pos[sk_labels == j]
        var_j = np.mean(np.sum((cluster_points - sk_centers[j])**2, axis=1))
        sk_variances.append(var_j)
        sk_std_devs.append(np.sqrt(var_j))
        
    print("Sklearn final centers:")
    print(f"Cluster 1 center: {sk_centers[0]}, Variance: {sk_variances[0]:.6f}, Std Dev: {sk_std_devs[0]:.6f}, Points: {np.sum(sk_labels == 0)}")
    print(f"Cluster 2 center: {sk_centers[1]}, Variance: {sk_variances[1]:.6f}, Std Dev: {sk_std_devs[1]:.6f}, Points: {np.sum(sk_labels == 1)}")
except ImportError:
    print("\nscikit-learn is not installed or import failed.")
