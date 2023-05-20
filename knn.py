import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn(X_train, y_train, X_test, k):
    predictions = []
    
    for test_point in X_test:
        distances = []
        
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        
        k_nearest = distances[:k]
        labels = [label for _, label in k_nearest]
        
        prediction = max(set(labels), key=labels.count)
        predictions.append(prediction)
    
    return predictions


# User input for training data
n_train = int(input("Enter the number of training samples (points): "))
X_train = np.zeros((n_train, 2))
y_train = np.zeros(n_train)

print("Enter the training points (One per line, e.g., 1 2, then press Enter, 5 6, etc.):")
for i in range(n_train):
    x, y = map(float, input().split())
    X_train[i] = [x, y]

print("Enter the labels for each training point (One per line):")
for i in range(n_train):
    y_train[i] = int(input())

# User input for test data
n_test = int(input("Enter the number of points to test: "))
X_test = np.zeros((n_test, 2))

print("Enter the points to test (One per line):")
for i in range(n_test):
    x, y = map(float, input().split())
    X_test[i] = [x, y]

# Set the value of k (number of neighbors)
k = int(input("Enter the value of k: "))

# Run KNN algorithm
predictions = knn(X_train, y_train, X_test, k)

# Output predictions
print("Predictions:")
for i, prediction in enumerate(predictions):
    print(f"Point {i+1} belongs to class: {prediction}")
