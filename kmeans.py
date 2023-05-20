import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def kmeans(X, k, max_iterations=100):
    centroids = X[np.random.choice(range(X.shape[0]), size=k, replace=False)]
    iterations = 0

    while iterations < max_iterations:
        clusters = [[] for _ in range(k)]
        for i in range(X.shape[0]):
            distances = [euclidean_distance(X[i], centroid) for centroid in centroids]
            closest_centroid_index = np.argmin(distances)
            clusters[closest_centroid_index].append(X[i])

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            new_centroids[i] = np.mean(clusters[i], axis=0)

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        iterations += 1

    return centroids, clusters

n_points = int(input("Enter the number of data points: "))
X = np.zeros((n_points, 2))

print("Enter the data points (one per line):")
for i in range(n_points):
    x, y = map(float, input().split())
    X[i] = [x, y]

k = int(input("Enter the value of k: "))

centroids, clusters = kmeans(X, k)

print("\nCluster centroids:")
for i, centroid in enumerate(centroids):
    print(f"Centroid {i+1}: {centroid}")

print("\nData points in each cluster:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {cluster}")
