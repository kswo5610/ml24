import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Step 1: Load the MNIST dataset
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"].astype(np.int64)

# Step 2: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Scale the data (KNN performs better with scaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Set up the KNeighborsClassifier and GridSearchCV
knn_clf = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 4, 5, 6],
    'weights': ['uniform', 'distance']
}

# Step 5: Perform GridSearch to find best hyperparameters
grid_search = GridSearchCV(knn_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Step 6: Evaluate the best model on the test set
best_knn_clf = grid_search.best_estimator_
y_test_pred = best_knn_clf.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_test_pred)

# Output the best hyperparameters and the test accuracy
print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Test set accuracy: {accuracy:.4f}")

# Optional: Show some predictions
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap='binary', **options)
    plt.axis("off")

some_test_digits = X_test_scaled[:100]
plot_digits(some_test_digits)
plt.show()

