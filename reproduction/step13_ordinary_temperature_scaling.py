import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. RANDOM SEEDS
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. CREATE SYNTHETIC DATASET
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
# 3. DATA SPLITS
#
# 6000 training
# 2000 tuning
# 2000 conformal calibration
# 2000 test
# ============================================================

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    train_size=6000,
    stratify=y,
    random_state=42
)

X_tuning, X_remaining, y_tuning, y_remaining = train_test_split(
    X_remaining,
    y_remaining,
    train_size=2000,
    stratify=y_remaining,
    random_state=42
)

X_conformal, X_test, y_conformal, y_test = train_test_split(
    X_remaining,
    y_remaining,
    train_size=2000,
    stratify=y_remaining,
    random_state=42
)


print("\n================ SPLITS ================")

print("Training:", X_train.shape, y_train.shape)
print("Tuning:", X_tuning.shape, y_tuning.shape)
print("Conformal calibration:", X_conformal.shape, y_conformal.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# 4. STANDARDIZE FEATURES
#
# IMPORTANT:
# scaler is fitted ONLY on training data
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_tuning = scaler.transform(X_tuning)
X_conformal = scaler.transform(X_conformal)
X_test = scaler.transform(X_test)


# ============================================================
# 5. CONVERT TO PYTORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_tuning_tensor = torch.tensor(
    X_tuning,
    dtype=torch.float32
)

y_tuning_tensor = torch.tensor(
    y_tuning,
    dtype=torch.long
)

X_conformal_tensor = torch.tensor(
    X_conformal,
    dtype=torch.float32
)

y_conformal_tensor = torch.tensor(
    y_conformal,
    dtype=torch.long
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.long
)


# ============================================================
# 6. TINY NEURAL NETWORK
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

epochs = 100


print("\n================ TRAINING ================")

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    logits = model(
        X_train_tensor
    )

    loss = criterion(
        logits,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()


    if (epoch + 1) % 20 == 0:

        predictions = torch.argmax(
            logits,
            dim=1
        )

        accuracy = (
            predictions == y_train_tensor
        ).float().mean().item()

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# 8. GET LOGITS
#
# We need:
#
# tuning logits     -> choose temperature
# conformal logits  -> calculate APS threshold
# test logits       -> final evaluation
# ============================================================

model.eval()

with torch.no_grad():

    tuning_logits = model(
        X_tuning_tensor
    )

    conformal_logits = model(
        X_conformal_tensor
    )

    test_logits = model(
        X_test_tensor
    )


print("\n================ LOGIT SHAPES ================")

print("Tuning logits:", tuning_logits.shape)
print("Conformal logits:", conformal_logits.shape)
print("Test logits:", test_logits.shape)


# ============================================================
# 9. NLL FUNCTION
#
# Ordinary Temperature Scaling chooses T by NLL.
#
# logits
#   ↓ divide by T
# scaled logits
#   ↓ cross entropy
# NLL
# ============================================================

def calculate_nll(logits, labels, temperature):

    scaled_logits = logits / temperature

    nll = F.cross_entropy(
        scaled_logits,
        labels
    )

    return nll.item()


# ============================================================
# 10. SEARCH FOR BEST TEMPERATURE
# ============================================================

temperatures = np.arange(
    0.30,
    2.01,
    0.05
)

nll_values = []


print("\n================ TEMPERATURE SCALING SEARCH ================")

print("T       Tuning NLL")
print("-------------------")


for T in temperatures:

    nll = calculate_nll(
        tuning_logits,
        y_tuning_tensor,
        T
    )

    nll_values.append(nll)

    print(
        f"{T:.2f}    {nll:.6f}"
    )


# ============================================================
# 11. SELECT TEMPERATURE WITH LOWEST NLL
# ============================================================

best_index = np.argmin(
    nll_values
)

T_TS = float(
    temperatures[best_index]
)

best_nll = nll_values[
    best_index
]


print("\n================ BEST TEMPERATURE ================")

print(
    f"Ordinary Temperature Scaling T: {T_TS:.2f}"
)

print(
    f"Minimum tuning NLL: {best_nll:.6f}"
)


# ============================================================
# 12. ECE FUNCTION
#
# ECE asks:
#
# When model confidence is X,
# is accuracy also approximately X?
# ============================================================

def calculate_ece(
        logits,
        labels,
        temperature=1.0,
        n_bins=10
):

    probabilities = F.softmax(
        logits / temperature,
        dim=1
    )

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    correct = (
        predictions == labels
    ).float()

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

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

        number_in_bin = in_bin.sum().item()

        if number_in_bin > 0:

            bin_accuracy = correct[
                in_bin
            ].mean()

            bin_confidence = confidences[
                in_bin
            ].mean()

            bin_fraction = in_bin.float().mean()

            ece += (
                bin_fraction
                *
                torch.abs(
                    bin_accuracy
                    -
                    bin_confidence
                )
            )

    return ece.item()


# ============================================================
# 13. NON-RANDOMIZED APS CALIBRATION SCORES
#
# Example:
#
# probabilities:
#
# class A = 0.50
# class B = 0.25
# class C = 0.15
#
# cumulative:
#
# A     = 0.50
# B     = 0.75
# C     = 0.90
#
# If true class = B:
#
# APS score = 0.75
# ============================================================

def calculate_aps_true_label_scores(
        logits,
        labels,
        temperature
):

    probabilities = F.softmax(
        logits / temperature,
        dim=1
    )

    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=1
    )

    # Find where the true label appears
    # in the sorted class list.
    true_label_mask = (
        sorted_indices
        ==
        labels.unsqueeze(1)
    )

    # Pick cumulative probability
    # corresponding to true label.
    true_label_scores = cumulative_probabilities[
        true_label_mask
    ]

    return true_label_scores


# ============================================================
# 14. CONFORMAL THRESHOLD
#
# alpha = 0.1
#
# target coverage ≈ 90%
# ============================================================

def calculate_conformal_threshold(
        scores,
        alpha=0.1
):

    n = len(scores)

    sorted_scores, _ = torch.sort(
        scores
    )

    # Finite-sample conformal quantile:
    #
    # ceil((n + 1) * (1-alpha))
    #
    rank = int(
        np.ceil(
            (n + 1)
            *
            (1 - alpha)
        )
    )

    # Python indexing starts from 0
    index = rank - 1

    # Safety in case index becomes too large
    index = min(
        index,
        n - 1
    )

    threshold = sorted_scores[
        index
    ].item()

    return threshold


# ============================================================
# 15. APS TEST EVALUATION
#
# For every possible class:
#
# calculate its cumulative APS score
#
# include class when:
#
# APS score <= conformal threshold
# ============================================================

def evaluate_aps(
        logits,
        labels,
        temperature,
        threshold
):

    probabilities = F.softmax(
        logits / temperature,
        dim=1
    )

    # Sort classes from highest probability
    # to lowest probability.
    sorted_probabilities, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    # APS cumulative probabilities.
    cumulative_probabilities = torch.cumsum(
        sorted_probabilities,
        dim=1
    )

    # cumulative_probabilities are currently
    # in SORTED class order.
    #
    # We now put the APS scores back into
    # their original class positions.
    candidate_scores = torch.zeros_like(
        cumulative_probabilities
    )

    candidate_scores.scatter_(
        1,
        sorted_indices,
        cumulative_probabilities
    )

    # Include labels whose APS score
    # is below the conformal threshold.
    prediction_sets = (
        candidate_scores
        <=
        threshold
    )


    # --------------------------------------------------------
    # COVERAGE
    #
    # Was the real label included?
    # --------------------------------------------------------

    row_indices = torch.arange(
        len(labels)
    )

    true_label_in_set = prediction_sets[
        row_indices,
        labels
    ]

    coverage = true_label_in_set.float().mean().item()


    # --------------------------------------------------------
    # AVERAGE SET SIZE
    # --------------------------------------------------------

    set_sizes = prediction_sets.sum(
        dim=1
    )

    average_set_size = set_sizes.float().mean().item()


    return (
        coverage,
        average_set_size,
        prediction_sets,
        probabilities
    )


# ============================================================
# 16. FUNCTION TO EVALUATE ONE TEMPERATURE
#
# This performs the whole conformal pipeline:
#
# T
# ↓
# conformal probabilities
# ↓
# APS calibration scores
# ↓
# threshold
# ↓
# test prediction sets
# ============================================================

def evaluate_temperature(
        temperature,
        alpha=0.1
):

    # --------------------------------------------------------
    # ECE on TEST DATA
    # --------------------------------------------------------

    ece = calculate_ece(
        test_logits,
        y_test_tensor,
        temperature=temperature,
        n_bins=10
    )


    # --------------------------------------------------------
    # APS scores on CONFORMAL CALIBRATION DATA
    # --------------------------------------------------------

    calibration_scores = calculate_aps_true_label_scores(
        conformal_logits,
        y_conformal_tensor,
        temperature
    )


    # --------------------------------------------------------
    # Calculate conformal threshold
    # --------------------------------------------------------

    threshold = calculate_conformal_threshold(
        calibration_scores,
        alpha=alpha
    )


    # --------------------------------------------------------
    # Evaluate APS on TEST DATA
    # --------------------------------------------------------

    (
        coverage,
        average_set_size,
        prediction_sets,
        probabilities
    ) = evaluate_aps(
        test_logits,
        y_test_tensor,
        temperature,
        threshold
    )


    return {
        "temperature": temperature,
        "ece": ece,
        "threshold": threshold,
        "coverage": coverage,
        "average_set_size": average_set_size,
        "prediction_sets": prediction_sets,
        "probabilities": probabilities
    }


# ============================================================
# 17. BASELINE
#
# Baseline means:
#
# T = 1
# ============================================================

baseline_results = evaluate_temperature(
    temperature=1.0,
    alpha=0.1
)


# ============================================================
# 18. ORDINARY TEMPERATURE SCALING
#
# Use T selected by minimum TUNING NLL
# ============================================================

ts_results = evaluate_temperature(
    temperature=T_TS,
    alpha=0.1
)


# ============================================================
# 19. FINAL COMPARISON
# ============================================================

print("\n================ FINAL COMPARISON ================")

print(
    f"{'Method':<25}"
    f"{'T':>8}"
    f"{'ECE':>12}"
    f"{'Coverage':>12}"
    f"{'Avg APS Size':>15}"
)

print("-" * 72)


print(
    f"{'Baseline':<25}"
    f"{baseline_results['temperature']:>8.2f}"
    f"{baseline_results['ece']:>12.4f}"
    f"{baseline_results['coverage']:>12.4f}"
    f"{baseline_results['average_set_size']:>15.4f}"
)


print(
    f"{'Temperature Scaling':<25}"
    f"{ts_results['temperature']:>8.2f}"
    f"{ts_results['ece']:>12.4f}"
    f"{ts_results['coverage']:>12.4f}"
    f"{ts_results['average_set_size']:>15.4f}"
)


# ============================================================
# 20. SHOW CONFORMAL THRESHOLDS
# ============================================================

print("\n================ APS THRESHOLDS ================")

print(
    f"Baseline threshold (T=1.00): "
    f"{baseline_results['threshold']:.6f}"
)

print(
    f"TS threshold (T={T_TS:.2f}): "
    f"{ts_results['threshold']:.6f}"
)


# ============================================================
# 21. INSPECT A FEW TEST SAMPLES
# ============================================================

print("\n================ SAMPLE TEST PREDICTIONS ================")

for i in range(3):

    true_class = y_test_tensor[i].item()

    baseline_probs = baseline_results[
        "probabilities"
    ][i]

    ts_probs = ts_results[
        "probabilities"
    ][i]

    baseline_set = torch.where(
        baseline_results[
            "prediction_sets"
        ][i]
    )[0].tolist()

    ts_set = torch.where(
        ts_results[
            "prediction_sets"
        ][i]
    )[0].tolist()


    print(
        f"\nSample {i + 1}"
    )

    print(
        "True class:",
        true_class
    )

    print(
        "Baseline max confidence:",
        f"{baseline_probs.max().item():.4f}"
    )

    print(
        "TS max confidence:",
        f"{ts_probs.max().item():.4f}"
    )

    print(
        "Baseline APS set:",
        baseline_set
    )

    print(
        "TS APS set:",
        ts_set
    )


# ============================================================
# 22. SIMPLE INTERPRETATION
# ============================================================

print("\n================ CHANGE AFTER TEMPERATURE SCALING ================")

ece_change = (
    ts_results["ece"]
    -
    baseline_results["ece"]
)

size_change = (
    ts_results["average_set_size"]
    -
    baseline_results["average_set_size"]
)


if ece_change < 0:

    print(
        "ECE decreased -> calibration improved."
    )

else:

    print(
        "ECE increased -> calibration became worse."
    )


if size_change < 0:

    print(
        "APS average set size decreased."
    )

elif size_change > 0:

    print(
        "APS average set size increased."
    )

else:

    print(
        "APS average set size stayed the same."
    )