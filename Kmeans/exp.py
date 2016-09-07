from sklearn import cluster
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import euclidean


def proba(centroids, points):
    results = np.zeros(shape=(len(points),2), dtype=float)
    for i in range(len(points)):
        for centroid in centroids:
            print points[i], centroid[0], centroid[1]
            results[i][centroid[1]] += euclidean(points[i], centroid[0])
    results /= results.sum(axis=1).reshape(results.shape[0], 1)
    return np.ones(shape=results.shape, dtype=float) - results


x = np.array([[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]])
y = np.array([0,0,0,1,1,1])

# set number of clusters
clusters_count = 2

# train
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(x)

# find out points in each cluster
Kclusters = k_means.labels_
centroids = []
cluster_indices = []
for j in range(clusters_count):
    indices = [i for i in range(len(x)) if Kclusters[i] == j]
    centroids.append([x[indices,:].mean(axis=0), mode(y[indices])[0][0]])

print centroids
print proba(centroids, [[0,0],[0,0.5],[0,1],[1,1],[0.5,0],[1,0]])

