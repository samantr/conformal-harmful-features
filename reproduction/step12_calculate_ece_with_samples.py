import numpy as np
import torch
import torch.nn as nn

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. CREATE DATASET
# ============================================================

X, y = make_classification(
    n_samples=12000,
    n_features=20,
    n_informative=18,
    n_redundant=2,
    n_classes=20,
    n_clusters_per_class=1,
    class_sep=2.0,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3. CREATE DATA SPLITS
# ============================================================

# First:
# 6000 training
# 6000 remaining

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=6000,
    stratify=y,
    random_state=42
)


# From remaining 6000:
# 2000 tuning
# 4000 remaining

X_tune, X_remaining, y_tune, y_remaining = train_test_split(
    X_remaining,
    y_remaining,
    test_size=4000,
    stratify=y_remaining,
    random_state=42
)


# From remaining 4000:
# 2000 conformal calibration
# 2000 test

X_conf, X_test, y_conf, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=2000,
    stratify=y_remaining,
    random_state=42
)


print()
print("================ SPLITS ================")

print("Training:", X_train.shape, y_train.shape)
print("Tuning:", X_tune.shape, y_tune.shape)
print("Conformal calibration:", X_conf.shape, y_conf.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# 4. SCALE FEATURES
# ============================================================

# Learn scaling ONLY from training data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_tune = scaler.transform(X_tune)
X_conf = scaler.transform(X_conf)
X_test = scaler.transform(X_test)


# ============================================================
# 5. CONVERT TRAINING DATA TO PYTORCH
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
# 6. CREATE SMALL NEURAL NETWORK
# ============================================================

class TinyClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 20)
        )

    def forward(self, x):
        return self.network(x)


model = TinyClassifier()


# ============================================================
# 7. TRAIN MODEL
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


print()
print("================ TRAINING ================")


for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train_tensor)

    loss = criterion(
        logits,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()


    # Print every 20 epochs
    if (epoch + 1) % 20 == 0:

        predictions = torch.argmax(
            logits,
            dim=1
        )

        accuracy = (
            predictions == y_train_tensor
        ).float().mean()

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.4f}"
        )


# ============================================================
# 8. GET TEST PROBABILITIES
# ============================================================

model.eval()

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


with torch.no_grad():

    # Raw model output
    logits_test = model(X_test_tensor)

    # IMPORTANT:
    # No temperature scaling here.
    #
    # This is equivalent to:
    #
    # T = 1.0

    probabilities = torch.softmax(
        logits_test,
        dim=1
    ).numpy()


print()
print("================ TEST PROBABILITIES ================")

print("Probability matrix shape:", probabilities.shape)


# Show first test sample
print()
print("First test sample probabilities:")

for class_index, probability in enumerate(probabilities[0]):

    print(
        f"Class {class_index:2d}: "
        f"{probability:.4f}"
    )


# ============================================================
# 9. GET PREDICTION + CONFIDENCE + CORRECTNESS
# ============================================================

# Which class has the highest probability?
predictions = np.argmax(
    probabilities,
    axis=1
)


# What is that highest probability?
confidences = np.max(
    probabilities,
    axis=1
)


# Was the prediction correct?
#
# Correct = 1
# Wrong   = 0

correct = (
    predictions == y_test
).astype(float)


print()
print("================ FIRST TEST EXAMPLES ================")

for i in range(10):

    print(
        f"Sample {i:2d} | "
        f"True={y_test[i]:2d} | "
        f"Predicted={predictions[i]:2d} | "
        f"Confidence={confidences[i]:.4f} | "
        f"Correct={int(correct[i])}"
    )


# ============================================================
# 10. ECE FUNCTION
# ============================================================

def calculate_ece(
        confidences,
        correct,
        n_bins=10
):

    # Example with 10 bins:
    #
    # 0.0 - 0.1
    # 0.1 - 0.2
    # ...
    # 0.9 - 1.0

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )


    ece = 0.0


    print()
    print("================ ECE BINS ================")

    print(
        "Bin       | Samples | "
        "Avg Confidence | Accuracy | Gap"
    )

    print("-" * 62)


    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]


        # Find samples belonging to this confidence bin
        if i == 0:

            in_bin = (
                (confidences >= lower)
                &
                (confidences <= upper)
            )

        else:

            in_bin = (
                (confidences > lower)
                &
                (confidences <= upper)
            )


        count = np.sum(in_bin)


        # Empty bin
        if count == 0:

            print(
                f"{lower:.1f}-{upper:.1f}   | "
                f"{0:7d} | "
                f"{'-':14} | "
                f"{'-':8} | "
                f"{'-'}"
            )

            continue


        # Average confidence in this bin
        bin_confidence = np.mean(
            confidences[in_bin]
        )


        # Actual accuracy in this bin
        bin_accuracy = np.mean(
            correct[in_bin]
        )


        # Calibration error for this bin
        difference = abs(
            bin_accuracy - bin_confidence
        )


        # How important is this bin?
        weight = count / len(confidences)


        # Add weighted gap to ECE
        ece += weight * difference


        print(
            f"{lower:.1f}-{upper:.1f}   | "
            f"{count:7d} | "
            f"{bin_confidence:14.4f} | "
            f"{bin_accuracy:8.4f} | "
            f"{difference:.4f}"
        )


    return ece


# ============================================================
# 11. CALCULATE BASELINE ECE
# ============================================================

ece = calculate_ece(
    confidences,
    correct,
    n_bins=10
)


# ============================================================
# 12. FINAL RESULTS
# ============================================================

test_accuracy = np.mean(correct)

average_confidence = np.mean(confidences)


print()
print("================ CALIBRATION RESULTS ================")

print("Temperature:        1.0000")
print(f"Test accuracy:      {test_accuracy:.4f}")
print(f"Average confidence: {average_confidence:.4f}")
print(f"ECE:                {ece:.4f}")


# ============================================================
# 13. SIMPLE INTERPRETATION
# ============================================================

print()
print("================ SIMPLE INTERPRETATION ================")


if average_confidence > test_accuracy:

    print(
        "Overall clue: the model appears OVERCONFIDENT."
    )

    print(
        "Its average confidence is higher "
        "than its actual accuracy."
    )


elif average_confidence < test_accuracy:

    print(
        "Overall clue: the model appears UNDERCONFIDENT."
    )

    print(
        "Its average confidence is lower "
        "than its actual accuracy."
    )


else:

    print(
        "Average confidence and accuracy are equal."
    )


print()
print(
    "Remember: ECE is more useful than only comparing "
    "overall confidence and accuracy,"
)

print(
    "because ECE compares them separately "
    "inside confidence bins."
)