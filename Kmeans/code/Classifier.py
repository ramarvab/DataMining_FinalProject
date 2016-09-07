from sklearn import cluster
import numpy as np
from scipy.stats import mode
from scipy.spatial.distance import euclidean


class Classifier(object):

    @staticmethod
    def proba(centroids, points):
        results = np.zeros(shape=(len(points),2), dtype=float)
        for i in range(len(points)):
            for centroid in centroids:
                results[i][centroid[1]] += euclidean(points[i], centroid[0])
        results /= results.sum(axis=1).reshape(results.shape[0], 1)
        return results

    @staticmethod
    def confidence_table(train_data, train_class, test_data):
        # set number of clusters
        clusters_count = 25

        # train
        k_means = cluster.KMeans(n_clusters=2)
        k_means.fit(train_data)

        # find out points in each cluster
        Kclusters = k_means.labels_
        centroids = []
        cluster_indices = []
        for j in range(clusters_count):
            indices = [i for i in range(len(train_data)) if Kclusters[i] == j]
            if indices:
                centroids.append([train_data[indices,:].mean(axis=0), mode(train_class[indices])[0][0]])

        return Classifier.proba(centroids, test_data)