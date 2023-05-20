import numpy as np

def calculate_eigen(X):
    n = X.shape[0]
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))

    for i in range(n):
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)

        for _ in range(100):
            v_new = np.dot(X, v)
            v_new = v_new / np.linalg.norm(v_new)

            if np.linalg.norm(v_new - v) < 1e-6:
                break

            v = v_new

        eigenvalues[i] = np.dot(v, np.dot(X, v))
        eigenvectors[:, i] = v

        X = X - eigenvalues[i] * np.outer(v, v)

    return eigenvalues, eigenvectors


def pca(X, k):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    eigenvalues, eigenvectors = calculate_eigen(covariance_matrix)

    indices = np.argsort(eigenvalues)[::-1]
    selected_eigenvectors = eigenvectors[:, indices[:k]]

    reduced_data = np.dot(X_centered, selected_eigenvectors)

    return reduced_data


num_samples = int(input("Enter the number of samples: "))
num_features = int(input("Enter the number of features: "))

print("Data matrix:")
data = np.zeros((num_samples, num_features))
for i in range(num_samples):
    for j in range(num_features):
        data[i][j] = float(input(f"Enter value for data[{i}][{j}]: "))

k = int(input("Enter the number of principal components (k): "))
reduced_data = pca(data, k)

print("Original data shape:", data.shape)
print("Reduced data shape:", reduced_data.shape)
print("Reduced data:")
print(reduced_data)
