from sklearn.datasets import make_classification
import numpy as np

# Create our artificial classification dataset
X, y = make_classification(
    n_samples=6000,
    n_features=10,
    n_informative=8,
    n_redundant=0,
    n_classes=5,
    n_clusters_per_class=1,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst sample features:")
print(X[0])

print("\nFirst sample class:")
print(y[0])

print("\nFirst 10 classes:")
print(y[:10])

# Count how many samples belong to each class
classes, counts = np.unique(y, return_counts=True)

print("\nClass counts:")
for class_number, count in zip(classes, counts):
    print(f"Class {class_number}: {count}")