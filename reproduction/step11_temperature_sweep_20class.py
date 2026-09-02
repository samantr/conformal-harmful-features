import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. Create Synthetic Dataset v2
#
# Harder than the original 5-class toy dataset:
# - more classes
# - more features
# - slight label noise
#
# We choose these settings ONCE before seeing the result.
# ============================================================

X, y = make_classification(
    n_samples=12000,
    n_features=20,
    n_informative=15,
    n_redundant=0,
    n_classes=20,
    n_clusters_per_class=1,
    class_sep=1.0,
    flip_y=0.02,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3. Create the SAME splits as Step 10
#
# 6000 training
# 2000 tuning
# 2000 conformal calibration
# 2000 test
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=6000,
    random_state=42,
    stratify=y
)

X_tuning, X_rest, y_tuning, y_rest = train_test_split(
    X_temp,
    y_temp,
    test_size=4000,
    random_state=42,
    stratify=y_temp
)

X_conformal, X_test, y_conformal, y_test = train_test_split(
    X_rest,
    y_rest,
    test_size=2000,
    random_state=42,
    stratify=y_rest
)


print("\n================ SPLITS ================")
print("Training:", X_train.shape, y_train.shape)
print("Tuning:", X_tuning.shape, y_tuning.shape)
print("Conformal calibration:", X_conformal.shape, y_conformal.shape)
print("Test:", X_test.shape, y_test.shape)


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
# 5. Tiny neural network for the 20-class dataset
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
# 6. Train the classifier
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 120


print("\n================ TRAINING ================")

for epoch in range(1, epochs + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train_tensor)

    loss = criterion(
        logits,
        y_train_tensor
    )

    loss.backward()

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
# 7. Get LOGITS for conformal and test data
#
# IMPORTANT CHANGE FROM STEP 10:
#
# Step 10:
# logits -> softmax once
#
# Step 11:
# logits -> divide by T -> softmax
#
# We therefore keep the logits here.
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


print("\n================ LOGITS ================")

print(
    "Conformal logits shape:",
    conformal_logits.shape
)

print(
    "Test logits shape:",
    test_logits.shape
)


# ============================================================
# 7B. Inspect how good/confident the new classifier is
# ============================================================

with torch.no_grad():

    conformal_predictions = torch.argmax(
        conformal_logits,
        dim=1
    ).numpy()

    test_predictions = torch.argmax(
        test_logits,
        dim=1
    ).numpy()

    test_probs_T1 = torch.softmax(
        test_logits.double(),
        dim=1
    ).numpy()


conformal_accuracy = np.mean(
    conformal_predictions == y_conformal
)

test_accuracy = np.mean(
    test_predictions == y_test
)

average_max_probability = np.mean(
    np.max(test_probs_T1, axis=1)
)


print("\n================ MODEL DIAGNOSTICS ================")

print(
    f"Conformal accuracy: {conformal_accuracy:.4f}"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)

print(
    f"Average max probability at T=1.0: "
    f"{average_max_probability:.4f}"
)


# ============================================================
# 8. SAME non-randomized APS function as Step 10
# ============================================================

def aps_scores(probabilities):
    """
    Calculate the non-randomized APS score
    for EVERY class.
    """

    # Sort class indices from highest probability to lowest
    order = np.argsort(
        probabilities
    )[::-1]

    # Sort probabilities
    sorted_probs = probabilities[
        order
    ]

    # Cumulative probability
    cumulative_probs = np.cumsum(
        sorted_probs
    )

    # Put each cumulative score back
    # into its original class position
    scores = np.empty_like(
        probabilities
    )

    scores[order] = cumulative_probs

    return scores


# ============================================================
# 9. Settings for conformal prediction
# ============================================================

alpha = 0.10

target_coverage = 1 - alpha

temperatures = [
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3
]


# ============================================================
# 10. Evaluate ONE temperature
# ============================================================

def evaluate_temperature(T):

    # --------------------------------------------------------
    # A. TEMPERATURE SCALING
    #
    # logits / T -> softmax probabilities
    # --------------------------------------------------------

    with torch.no_grad():

        conformal_probs = torch.softmax(
            conformal_logits / T,
            dim=1
        ).numpy()

        test_probs = torch.softmax(
            test_logits / T,
            dim=1
        ).numpy()


    # --------------------------------------------------------
    # B. CALCULATE TRUE-LABEL APS SCORES
    #    on the conformal calibration set
    # --------------------------------------------------------

    calibration_scores = []

    for probs, true_class in zip(
        conformal_probs,
        y_conformal
    ):

        scores = aps_scores(
            probs
        )

        true_class_score = scores[
            true_class
        ]

        calibration_scores.append(
            true_class_score
        )


    calibration_scores = np.array(
        calibration_scores
    )


    # --------------------------------------------------------
    # C. CALCULATE A NEW TAU FOR THIS TEMPERATURE
    #
    # This is VERY important.
    #
    # Every T changes probabilities.
    # Therefore every T changes APS scores.
    # Therefore every T needs its own tau.
    # --------------------------------------------------------

    n = len(
        calibration_scores
    )

    rank = int(
        np.ceil(
            (n + 1)
            * (1 - alpha)
        )
    )

    rank = min(
        rank,
        n
    )

    sorted_calibration_scores = np.sort(
        calibration_scores
    )

    tau = sorted_calibration_scores[
        rank - 1
    ]


    # --------------------------------------------------------
    # D. GENERATE TEST PREDICTION SETS
    # --------------------------------------------------------

    prediction_sets = []

    for probs in test_probs:

        scores = aps_scores(
            probs
        )

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


    # --------------------------------------------------------
    # E. COVERAGE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # F. AVERAGE PREDICTION-SET SIZE
    # --------------------------------------------------------

    set_sizes = [
        len(prediction_set)
        for prediction_set
        in prediction_sets
    ]

    average_set_size = np.mean(
        set_sizes
    )


    # Extra diagnostic:
    # We already learned that non-randomized APS
    # can produce empty sets.
    empty_sets = set_sizes.count(
        0
    )


    return (
        tau,
        coverage,
        average_set_size,
        empty_sets
    )


# ============================================================
# 11. Tiny sanity check:
#     show how temperature changes ONE probability vector
# ============================================================

print("\n================ TEMPERATURE EFFECT ================")

sample_index = 0

for T in [0.5, 1.0, 1.3]:

    with torch.no_grad():

        probs = torch.softmax(
            test_logits[sample_index] / T,
            dim=0
        ).numpy()

    print(
        f"\nT = {T:.1f}"
    )

    print(
        "First test sample probabilities:",
        np.round(probs, 6)
    )


# ============================================================
# 12. RUN THE TEMPERATURE SWEEP
# ============================================================

results = []

average_sizes = []


print("\n================ TEMPERATURE SWEEP ================")

print(
    "T     Tau          Coverage     Avg Size     Empty Sets"
)

print(
    "--------------------------------------------------------"
)


for T in temperatures:

    (
        tau,
        coverage,
        average_set_size,
        empty_sets
    ) = evaluate_temperature(
        T
    )

    results.append(
        (
            T,
            tau,
            coverage,
            average_set_size,
            empty_sets
        )
    )

    average_sizes.append(
        average_set_size
    )

    print(
        f"{T:<4.1f}  "
        f"{tau:<12.8f} "
        f"{coverage:<12.4f} "
        f"{average_set_size:<12.4f} "
        f"{empty_sets}"
    )


# ============================================================
# 13. Highlight the baseline T = 1.0
# ============================================================

print("\n================ BASELINE ================")

for (
    T,
    tau,
    coverage,
    average_set_size,
    empty_sets
) in results:

    if T == 1.0:

        print(
            "T = 1.0 is the SAME temperature "
            "used implicitly in Step 10."
        )

        print(
            f"Tau: {tau:.8f}"
        )

        print(
            f"Coverage: {coverage:.4f}"
        )

        print(
            f"Average set size: "
            f"{average_set_size:.4f}"
        )

        print(
            f"Empty sets: "
            f"{empty_sets}"
        )


# ============================================================
# 14. Plot:
#
# Temperature vs Average Prediction-Set Size
# ============================================================

plt.figure()

plt.plot(
    temperatures,
    average_sizes,
    marker="o"
)

plt.xlabel(
    "Temperature"
)

plt.ylabel(
    "Average Prediction-Set Size"
)

plt.title(
    "Temperature vs APS Average Set Size — 20-Class Synthetic"
)

plt.grid(
    True
)

plt.show()
