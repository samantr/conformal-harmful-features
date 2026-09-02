from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


# Create a tiny dataset that we can visualize
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=3,
    n_clusters_per_class=1,
    class_sep=1.0,
    random_state=42
)


print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 samples:")
print(X[:5])

print("\nFirst 5 classes:")
print(y[:5])


# Plot each class separately
for class_number in range(3):

    # Select only samples belonging to this class
    class_points = X[y == class_number]

    plt.scatter(
        class_points[:, 0],
        class_points[:, 1],
        label=f"Class {class_number}",
        alpha=0.6
    )


plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Tiny 2D Classification Dataset")
plt.legend()

plt.show()