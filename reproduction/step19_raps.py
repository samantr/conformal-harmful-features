import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
    class_sep=1.3,
    random_state=SEED
)

X = X.astype(np.float32)
y = y.astype(np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3. SPLITS
#
# 6000 training
# 2000 tuning
# 2000 conformal calibration
# 2000 test
# ============================================================

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.50,
    random_state=SEED,
    stratify=y
)

X_tune, X_remaining, y_tune, y_remaining = train_test_split(
    X_remaining,
    y_remaining,
    test_size=2 / 3,
    random_state=SEED,
    stratify=y_remaining
)

X_conf, X_test, y_conf, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.50,
    random_state=SEED,
    stratify=y_remaining
)

print("\n================ SPLITS ================")
print("Training:", X_train.shape, y_train.shape)
print("Tuning:", X_tune.shape, y_tune.shape)
print("Conformal calibration:", X_conf.shape, y_conf.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# 4. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_tune_tensor = torch.tensor(X_tune, dtype=torch.float32)
y_tune_tensor = torch.tensor(y_tune, dtype=torch.long)

X_conf_tensor = torch.tensor(X_conf, dtype=torch.float32)
y_conf_tensor = torch.tensor(y_conf, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# ============================================================
# 5. SMALL NEURAL NETWORK
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

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\n================ TRAINING ================")

epochs = 100

for epoch in range(1, epochs + 1):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train_tensor)

    loss = criterion(logits, y_train_tensor)

    loss.backward()

    optimizer.step()

    if epoch % 20 == 0:

        predictions = torch.argmax(logits, dim=1)

        accuracy = (
            predictions == y_train_tensor
        ).float().mean().item()

        print(
            f"Epoch {epoch:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# 7. GET LOGITS
# ============================================================

model.eval()

with torch.no_grad():

    tune_logits = model(X_tune_tensor)
    conf_logits = model(X_conf_tensor)
    test_logits = model(X_test_tensor)


# ============================================================
# 8. SOFTMAX
#
# For Step 19 we simply use T = 1.
#
# The tuning set stays in the pipeline because we will need it
# for Temperature Scaling / ConfTS comparisons later.
# ============================================================

TEMPERATURE = 1.0


def softmax_with_temperature(logits, temperature):

    return torch.softmax(
        logits / temperature,
        dim=1
    )


conf_probs = softmax_with_temperature(
    conf_logits,
    TEMPERATURE
).cpu().numpy()

test_probs = softmax_with_temperature(
    test_logits,
    TEMPERATURE
).cpu().numpy()


# ============================================================
# 9. SETTINGS
# ============================================================

ALPHA = 0.10

TARGET_COVERAGE = 1 - ALPHA

K_REG = 1

LAMBDA_REG = 0.001


print("\n================ SETTINGS ================")

print("Alpha:", ALPHA)

print(
    "Target coverage:",
    TARGET_COVERAGE
)

print("k_reg:", K_REG)

print(
    "lambda:",
    LAMBDA_REG
)


# ============================================================
# 10. RANDOMIZED APS SCORE
#
# Example:
#
# A = 0.60
# B = 0.20
# C = 0.10
# D = 0.06
#
# Candidate = D
# u = 0.5
#
# APS =
# 0.60 + 0.20 + 0.10 + 0.5 * 0.06
# ============================================================

def randomized_aps_score(
    probabilities,
    label,
    u
):

    # Sort class indexes from highest probability
    # to lowest probability.
    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    # Sorted probabilities.
    sorted_probs = probabilities[
        sorted_indices
    ]

    # Find where our candidate label appears.
    rank_index = np.where(
        sorted_indices == label
    )[0][0]

    # Everything BEFORE the candidate is
    # included completely.
    previous_sum = sorted_probs[
        :rank_index
    ].sum()

    # Candidate itself is multiplied by u.
    randomized_part = (
        u * sorted_probs[rank_index]
    )

    score = (
        previous_sum
        + randomized_part
    )

    return score


# ============================================================
# 11. RANDOMIZED RAPS SCORE
#
# RAPS =
#
# randomized APS
# +
# lambda * max(rank - k_reg, 0)
#
# IMPORTANT:
#
# rank_index is:
#
# 0, 1, 2, 3...
#
# But human rank is:
#
# 1, 2, 3, 4...
#
# Therefore:
#
# rank = rank_index + 1
# ============================================================

def randomized_raps_score(
    probabilities,
    label,
    u,
    k_reg=1,
    lambda_reg=0.001
):

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    sorted_probs = probabilities[
        sorted_indices
    ]

    rank_index = np.where(
        sorted_indices == label
    )[0][0]

    previous_sum = sorted_probs[
        :rank_index
    ].sum()

    randomized_part = (
        u * sorted_probs[rank_index]
    )

    # Convert:
    #
    # index 0 -> rank 1
    # index 1 -> rank 2
    # etc.
    rank = rank_index + 1

    penalty = (
        lambda_reg
        * max(
            rank - k_reg,
            0
        )
    )

    score = (
        previous_sum
        + randomized_part
        + penalty
    )

    return score


# ============================================================
# 12. SHOW PENALTIES BY RANK
# ============================================================

print(
    "\n================ RAPS PENALTY EXAMPLE ================"
)

for rank in range(1, 11):

    penalty = (
        LAMBDA_REG
        * max(
            rank - K_REG,
            0
        )
    )

    print(
        f"Rank {rank:2d} -> "
        f"Penalty = {penalty:.4f}"
    )


# ============================================================
# 13. CREATE RANDOM u VALUES
#
# We generate them ONCE.
#
# APS and RAPS use the SAME u values.
#
# This makes the comparison fair.
# ============================================================

rng = np.random.default_rng(SEED)

u_conf = rng.uniform(
    0.0,
    1.0,
    size=len(y_conf)
)

u_test = rng.uniform(
    0.0,
    1.0,
    size=len(y_test)
)


# ============================================================
# 14. APS CALIBRATION SCORES
# ============================================================

aps_calibration_scores = []

for i in range(
    len(y_conf)
):

    probabilities = conf_probs[i]

    true_label = y_conf[i]

    u = u_conf[i]

    score = randomized_aps_score(
        probabilities,
        true_label,
        u
    )

    aps_calibration_scores.append(
        score
    )

aps_calibration_scores = np.array(
    aps_calibration_scores
)


# ============================================================
# 15. RAPS CALIBRATION SCORES
# ============================================================

raps_calibration_scores = []

for i in range(
    len(y_conf)
):

    probabilities = conf_probs[i]

    true_label = y_conf[i]

    u = u_conf[i]

    score = randomized_raps_score(
        probabilities,
        true_label,
        u,
        k_reg=K_REG,
        lambda_reg=LAMBDA_REG
    )

    raps_calibration_scores.append(
        score
    )

raps_calibration_scores = np.array(
    raps_calibration_scores
)


# ============================================================
# 16. CONFORMAL QUANTILE
#
# k = ceil((n + 1) * (1-alpha))
#
# Then select the k-th smallest score.
# ============================================================

def conformal_threshold(
    scores,
    alpha
):

    sorted_scores = np.sort(
        scores
    )

    n = len(
        sorted_scores
    )

    k = int(
        np.ceil(
            (n + 1)
            * (1 - alpha)
        )
    )

    # Python indexing starts at zero.
    index = k - 1

    # Safety in case k > n.
    index = min(
        index,
        n - 1
    )

    return sorted_scores[
        index
    ]


aps_tau = conformal_threshold(
    aps_calibration_scores,
    ALPHA
)

raps_tau = conformal_threshold(
    raps_calibration_scores,
    ALPHA
)


print(
    "\n================ THRESHOLDS ================"
)

print(
    f"APS tau : {aps_tau:.6f}"
)

print(
    f"RAPS tau: {raps_tau:.6f}"
)


# ============================================================
# 17. CREATE RANDOMIZED APS PREDICTION SET
# ============================================================

def aps_prediction_set(
    probabilities,
    tau,
    u
):

    prediction_set = []

    number_of_classes = len(
        probabilities
    )

    for candidate_label in range(
        number_of_classes
    ):

        score = randomized_aps_score(
            probabilities,
            candidate_label,
            u
        )

        if score <= tau:

            prediction_set.append(
                candidate_label
            )

    return prediction_set


# ============================================================
# 18. CREATE RANDOMIZED RAPS PREDICTION SET
# ============================================================

def raps_prediction_set(
    probabilities,
    tau,
    u,
    k_reg=1,
    lambda_reg=0.001
):

    prediction_set = []

    number_of_classes = len(
        probabilities
    )

    for candidate_label in range(
        number_of_classes
    ):

        score = randomized_raps_score(
            probabilities,
            candidate_label,
            u,
            k_reg=k_reg,
            lambda_reg=lambda_reg
        )

        if score <= tau:

            prediction_set.append(
                candidate_label
            )

    return prediction_set


# ============================================================
# 19. GENERATE APS AND RAPS SETS
# ============================================================

aps_sets = []

raps_sets = []

for i in range(
    len(y_test)
):

    probabilities = test_probs[i]

    u = u_test[i]

    aps_set = aps_prediction_set(
        probabilities,
        aps_tau,
        u
    )

    raps_set = raps_prediction_set(
        probabilities,
        raps_tau,
        u,
        k_reg=K_REG,
        lambda_reg=LAMBDA_REG
    )

    aps_sets.append(
        aps_set
    )

    raps_sets.append(
        raps_set
    )


# ============================================================
# 20. METRICS
# ============================================================

def calculate_coverage(
    prediction_sets,
    true_labels
):

    covered = 0

    for prediction_set, true_label in zip(
        prediction_sets,
        true_labels
    ):

        if true_label in prediction_set:

            covered += 1

    return (
        covered
        / len(true_labels)
    )


def calculate_average_size(
    prediction_sets
):

    sizes = [
        len(prediction_set)
        for prediction_set
        in prediction_sets
    ]

    return np.mean(
        sizes
    )


aps_coverage = calculate_coverage(
    aps_sets,
    y_test
)

aps_average_size = calculate_average_size(
    aps_sets
)

raps_coverage = calculate_coverage(
    raps_sets,
    y_test
)

raps_average_size = calculate_average_size(
    raps_sets
)


# ============================================================
# 21. MODEL TEST ACCURACY
# ============================================================

test_predictions = np.argmax(
    test_probs,
    axis=1
)

test_accuracy = np.mean(
    test_predictions
    == y_test
)


# ============================================================
# 22. FINAL RESULTS
# ============================================================

print(
    "\n================ MODEL ================"
)

print(
    f"Test accuracy: {test_accuracy:.4f}"
)


print(
    "\n================ FINAL RESULTS ================"
)

print(
    f"{'Method':<10}"
    f"{'Coverage':>12}"
    f"{'Avg Set Size':>18}"
)

print(
    "-" * 40
)

print(
    f"{'APS':<10}"
    f"{aps_coverage:>12.4f}"
    f"{aps_average_size:>18.4f}"
)

print(
    f"{'RAPS':<10}"
    f"{raps_coverage:>12.4f}"
    f"{raps_average_size:>18.4f}"
)


# ============================================================
# 23. INSPECT SOME INDIVIDUAL TEST SAMPLES
# ============================================================

print(
    "\n================ SAMPLE PREDICTIONS ================"
)

for i in range(5):

    probabilities = test_probs[i]

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    print(
        f"\n--------------- SAMPLE {i + 1} ---------------"
    )

    print(
        "True class:",
        y_test[i]
    )

    print(
        f"u = {u_test[i]:.4f}"
    )

    print(
        "\nTop 10 probabilities:"
    )

    for rank_index in range(10):

        class_index = sorted_indices[
            rank_index
        ]

        probability = probabilities[
            class_index
        ]

        rank = (
            rank_index + 1
        )

        penalty = (
            LAMBDA_REG
            * max(
                rank - K_REG,
                0
            )
        )

        print(
            f"Rank {rank:2d} | "
            f"Class {class_index:2d} | "
            f"Probability = {probability:.4f} | "
            f"RAPS penalty = {penalty:.4f}"
        )

    print(
        "\nAPS prediction set:",
        aps_sets[i]
    )

    print(
        "APS set size:",
        len(aps_sets[i])
    )

    print(
        "\nRAPS prediction set:",
        raps_sets[i]
    )

    print(
        "RAPS set size:",
        len(raps_sets[i])
    )

    print(
        "\nTrue label in APS:",
        y_test[i] in aps_sets[i]
    )

    print(
        "True label in RAPS:",
        y_test[i] in raps_sets[i]
    )