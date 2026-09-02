import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. Create the same synthetic dataset
# ============================================================

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

print("\nClass counts:")
for class_id, count in enumerate(np.bincount(y)):
    print(f"Class {class_id}: {count}")


# ============================================================
# 3. Split the data
#
# 3000 training
# 1000 tuning
# 1000 conformal calibration
# 1000 test
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=3000,
    random_state=42,
    stratify=y
)

X_tuning, X_rest, y_tuning, y_rest = train_test_split(
    X_temp,
    y_temp,
    test_size=2000,
    random_state=42,
    stratify=y_temp
)

X_conformal, X_test, y_conformal, y_test = train_test_split(
    X_rest,
    y_rest,
    test_size=1000,
    random_state=42,
    stratify=y_rest
)


print("\n================ SPLITS ================")

print("Training:")
print(X_train.shape, y_train.shape)
print("Class counts:", np.bincount(y_train))

print("\nTuning:")
print(X_tuning.shape, y_tuning.shape)
print("Class counts:", np.bincount(y_tuning))

print("\nConformal calibration:")
print(X_conformal.shape, y_conformal.shape)
print("Class counts:", np.bincount(y_conformal))

print("\nTest:")
print(X_test.shape, y_test.shape)
print("Class counts:", np.bincount(y_test))


# ============================================================
# 4. Convert training data to PyTorch tensors
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)


# ============================================================
# 5. Tiny neural network
#
# 10 inputs
# ↓
# 32 neurons
# ↓
# 32 neurons
# ↓
# 5 logits
# ============================================================

class TinyClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 5)
        )

    def forward(self, x):
        return self.network(x)


model = TinyClassifier()


# ============================================================
# 6. Train the classifier
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 200


print("\n================ TRAINING ================")

for epoch in range(1, epochs + 1):

    model.train()

    # Clear previous gradients
    optimizer.zero_grad()

    # Features -> logits
    logits = model(X_train_tensor)

    # Calculate training loss
    loss = criterion(logits, y_train_tensor)

    # Calculate gradients
    loss.backward()

    # Update neural network weights
    optimizer.step()

    if epoch % 20 == 0:

        predicted_classes = torch.argmax(
            logits,
            dim=1
        )

        accuracy = (
            predicted_classes == y_train_tensor
        ).float().mean()

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.4f}"
        )


# ============================================================
# 7. Calculate probabilities for conformal and test data
#
# IMPORTANT:
# We do NOT use the tuning set in Step 10.
# We will need it later for Temperature Scaling / ConfTS.
# ============================================================

model.eval()

X_conformal_tensor = torch.tensor(
    X_conformal,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


with torch.no_grad():

    conformal_logits = model(
        X_conformal_tensor
    )

    test_logits = model(
        X_test_tensor
    )

    conformal_probs = torch.softmax(
        conformal_logits,
        dim=1
    ).numpy()

    test_probs = torch.softmax(
        test_logits,
        dim=1
    ).numpy()


print("\n================ PROBABILITIES ================")

print(
    "Conformal probability shape:",
    conformal_probs.shape
)

print(
    "Test probability shape:",
    test_probs.shape
)


# ============================================================
# 8. Non-randomized APS function
# ============================================================

def aps_scores(probabilities):
    """
    Calculate the non-randomized APS score
    for EVERY class.

    Example probabilities:

        class 0 = 0.15
        class 1 = 0.50
        class 2 = 0.10
        class 3 = 0.20
        class 4 = 0.05

    Sorted:

        class 1 = 0.50
        class 3 = 0.20
        class 0 = 0.15
        class 2 = 0.10
        class 4 = 0.05

    APS scores:

        class 1 -> 0.50
        class 3 -> 0.70
        class 0 -> 0.85
        class 2 -> 0.95
        class 4 -> 1.00
    """

    # Get class indices sorted from
    # highest probability to lowest.
    order = np.argsort(probabilities)[::-1]

    # Sort the actual probabilities
    sorted_probs = probabilities[order]

    # Calculate cumulative probabilities
    cumulative_probs = np.cumsum(sorted_probs)

    # scores[class_id] should give us
    # the APS score for that class.
    scores = np.empty_like(probabilities)

    scores[order] = cumulative_probs

    return scores


# ============================================================
# 9. Tiny manual APS demonstration
# ============================================================

print("\n================ MANUAL APS EXAMPLE ================")

example_probs = np.array([
    0.15,
    0.50,
    0.10,
    0.20,
    0.05
])

example_scores = aps_scores(example_probs)

example_order = np.argsort(example_probs)[::-1]

for class_id in example_order:

    print(
        f"Class {class_id} | "
        f"Probability = {example_probs[class_id]:.2f} | "
        f"APS score = {example_scores[class_id]:.2f}"
    )


# ============================================================
# 10. Calculate conformal APS scores
#
# For each conformal sample:
#
# probabilities
#       ↓
# APS scores for all classes
#       ↓
# KEEP ONLY TRUE CLASS SCORE
# ============================================================

calibration_scores = []


for probs, true_class in zip(
    conformal_probs,
    y_conformal
):

    scores = aps_scores(probs)

    true_class_score = scores[
        true_class
    ]

    calibration_scores.append(
        true_class_score
    )


calibration_scores = np.array(
    calibration_scores
)


print(
    "\nNumber of calibration scores:",
    len(calibration_scores)
)


# ============================================================
# 11. Inspect FIRST conformal sample manually
# ============================================================

i = 0

probs = conformal_probs[i]

true_class = y_conformal[i]

scores = aps_scores(probs)

order = np.argsort(probs)[::-1]


print("\n================ FIRST CONFORMAL SAMPLE ================")

print("True class:", true_class)

print("\nSorted classes:")

for rank, class_id in enumerate(
    order,
    start=1
):

    print(
        f"Rank {rank} | "
        f"Class {class_id} | "
        f"Probability = {probs[class_id]:.6f} | "
        f"APS score = {scores[class_id]:.6f}"
    )


print(
    "\nTrue-class APS score:",
    scores[true_class]
)


# ============================================================
# 12. Calculate conformal threshold tau
# ============================================================

alpha = 0.10

target_coverage = 1 - alpha

n = len(calibration_scores)


# Formula:
#
# ceil((n + 1) * (1 - alpha))
#
rank = int(
    np.ceil(
        (n + 1) * (1 - alpha)
    )
)

# Safety in case rank > n
rank = min(rank, n)


sorted_calibration_scores = np.sort(
    calibration_scores
)


# Python starts indexes at 0,
# so rank 901 is index 900.
tau = sorted_calibration_scores[
    rank - 1
]


print("\n================ CONFORMAL THRESHOLD ================")

print("Alpha:", alpha)

print(
    "Target coverage:",
    target_coverage
)

print(
    "Number of conformal samples:",
    n
)

print(
    "Quantile rank:",
    rank
)

print(
    "APS threshold tau:",
    tau
)


# ============================================================
# 13. Generate APS prediction sets for TEST samples
#
# For each candidate class:
#
# APS score <= tau
#
#       ↓
#
# include that class
# ============================================================

prediction_sets = []


for probs in test_probs:

    # Calculate APS score
    # for all 5 possible classes.
    scores = aps_scores(probs)

    prediction_set = []

    for class_id in range(
        len(probs)
    ):

        if scores[class_id] <= tau:

            prediction_set.append(
                class_id
            )

    prediction_sets.append(
        prediction_set
    )


# ============================================================
# 14. Inspect first 10 test examples
# ============================================================

print("\n================ FIRST 10 TEST SAMPLES ================")


for i in range(10):

    probs = test_probs[i]

    scores = aps_scores(probs)

    order = np.argsort(
        probs
    )[::-1]

    print("\n----------------------------------")

    print(
        "Test sample:",
        i
    )

    print(
        "True class:",
        y_test[i]
    )

    print("\nSorted probabilities and APS scores:")

    for class_id in order:

        print(
            f"Class {class_id} | "
            f"Probability = {probs[class_id]:.6f} | "
            f"APS score = {scores[class_id]:.6f}"
        )

    print(
        "\nPrediction set:",
        prediction_sets[i]
    )

    print(
        "True class included:",
        y_test[i] in prediction_sets[i]
    )


# ============================================================
# 15. Calculate COVERAGE
# ============================================================

covered = 0


for true_class, prediction_set in zip(
    y_test,
    prediction_sets
):

    if true_class in prediction_set:

        covered += 1


coverage = covered / len(
    y_test
)


# ============================================================
# 16. Calculate prediction-set sizes
# ============================================================

set_sizes = [
    len(prediction_set)
    for prediction_set
    in prediction_sets
]


average_set_size = np.mean(
    set_sizes
)


# ============================================================
# 17. Final results
# ============================================================

print("\n================ FINAL RESULTS ================")

print(
    "Number of test samples:",
    len(y_test)
)

print(
    "True class covered:",
    covered
)

print(
    f"Coverage: {coverage:.4f}"
)

print(
    f"Target coverage: {target_coverage:.4f}"
)

print(
    f"Average prediction-set size: "
    f"{average_set_size:.4f}"
)


# ============================================================
# 18. Prediction-set size distribution
# ============================================================

print("\nPrediction-set size counts:")

for size in range(6):

    count = set_sizes.count(
        size
    )

    print(
        f"Size {size}: {count}"
    )