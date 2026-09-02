import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. DATASET
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
# 3. SPLITS
# ============================================================

# 6000 training, 6000 remaining
X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    test_size=0.5,
    stratify=y,
    random_state=42
)

# 2000 tuning, 4000 remaining
X_tune, X_remaining, y_tune, y_remaining = train_test_split(
    X_remaining,
    y_remaining,
    train_size=2000,
    stratify=y_remaining,
    random_state=42
)

# 2000 conformal, 2000 test
X_conformal, X_test, y_conformal, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.5,
    stratify=y_remaining,
    random_state=42
)

print("\n================ SPLITS ================")
print("Training:", X_train.shape, y_train.shape)
print("Tuning:", X_tune.shape, y_tune.shape)
print("Conformal calibration:", X_conformal.shape, y_conformal.shape)
print("Test:", X_test.shape, y_test.shape)


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_tune = scaler.transform(X_tune)
X_conformal = scaler.transform(X_conformal)
X_test = scaler.transform(X_test)


# ============================================================
# 5. PYTORCH TENSORS
# ============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

X_tune_t = torch.tensor(X_tune, dtype=torch.float32)
X_conformal_t = torch.tensor(X_conformal, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)


# ============================================================
# 6. MODEL
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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = nn.CrossEntropyLoss()


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\n================ TRAINING ================")

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    logits = model(X_train_t)

    loss = criterion(logits, y_train_t)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        predictions = torch.argmax(logits, dim=1)

        accuracy = (
            predictions == y_train_t
        ).float().mean().item()

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# 8. SAVE LOGITS
# ============================================================

model.eval()

with torch.no_grad():

    tune_logits = model(X_tune_t)
    conformal_logits = model(X_conformal_t)
    test_logits = model(X_test_t)


# ============================================================
# 9. SOFTMAX WITH TEMPERATURE
# ============================================================

def probabilities_from_logits(logits, temperature):

    return torch.softmax(
        logits / temperature,
        dim=1
    ).cpu().numpy()


# ============================================================
# 10. NON-RANDOMIZED APS SCORES
# ============================================================

def aps_score_matrix(probabilities):

    # Sort class probabilities from biggest to smallest
    order = np.argsort(
        -probabilities,
        axis=1
    )

    sorted_probabilities = np.take_along_axis(
        probabilities,
        order,
        axis=1
    )

    # APS cumulative probabilities
    cumulative = np.cumsum(
        sorted_probabilities,
        axis=1
    )

    # Put scores back into original class positions
    scores = np.empty_like(cumulative)

    rows = np.arange(
        len(probabilities)
    )[:, None]

    scores[rows, order] = cumulative

    return scores


# ============================================================
# 11. CONFORMAL THRESHOLD
# ============================================================

def conformal_threshold(true_scores, alpha=0.1):

    n = len(true_scores)

    k = int(
        np.ceil(
            (n + 1) * (1 - alpha)
        )
    )

    k = min(k, n)

    sorted_scores = np.sort(true_scores)

    return sorted_scores[k - 1]


# ============================================================
# 12. COVERAGE AND AVERAGE SET SIZE
# ============================================================

def evaluate_aps(
    conformal_logits,
    y_conformal,
    test_logits,
    y_test,
    temperature,
    alpha=0.1
):

    # ----- Conformal calibration -----

    conformal_probs = probabilities_from_logits(
        conformal_logits,
        temperature
    )

    conformal_scores = aps_score_matrix(
        conformal_probs
    )

    true_conformal_scores = conformal_scores[
        np.arange(len(y_conformal)),
        y_conformal
    ]

    tau = conformal_threshold(
        true_conformal_scores,
        alpha
    )

    # ----- Test -----

    test_probs = probabilities_from_logits(
        test_logits,
        temperature
    )

    test_scores = aps_score_matrix(
        test_probs
    )

    prediction_sets = (
        test_scores <= tau
    )

    coverage = np.mean(
        prediction_sets[
            np.arange(len(y_test)),
            y_test
        ]
    )

    average_size = np.mean(
        prediction_sets.sum(axis=1)
    )

    return coverage, average_size, tau


# ============================================================
# 13. ECE
# ============================================================

def calculate_ece(
    probabilities,
    labels,
    n_bins=10
):

    confidences = np.max(
        probabilities,
        axis=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    accuracies = (
        predictions == labels
    )

    bin_edges = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

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

        if np.sum(in_bin) == 0:
            continue

        bin_accuracy = np.mean(
            accuracies[in_bin]
        )

        bin_confidence = np.mean(
            confidences[in_bin]
        )

        bin_fraction = np.mean(
            in_bin
        )

        ece += (
            bin_fraction
            *
            abs(
                bin_accuracy
                -
                bin_confidence
            )
        )

    return ece


# ============================================================
# 14. TEMPERATURE CANDIDATES
# ============================================================

temperatures = np.array([
    0.1,
    0.2,
    0.3,
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
])


# ============================================================
# 15. ORDINARY TEMPERATURE SCALING
# ============================================================

print(
    "\n================ ORDINARY TEMPERATURE SCALING ================"
)

nll_results = []

y_tune_tensor = torch.tensor(
    y_tune,
    dtype=torch.long
)

for T in temperatures:

    nll = F.cross_entropy(
        tune_logits / T,
        y_tune_tensor
    ).item()

    nll_results.append(nll)

    print(
        f"T = {T:.1f} | "
        f"NLL = {nll:.6f}"
    )


best_ts_index = np.argmin(
    nll_results
)

T_TS = temperatures[
    best_ts_index
]

print(
    "\nOrdinary TS temperature:",
    T_TS
)


# ============================================================
# 16. SPLIT TUNING DATA FOR ConfTS
# ============================================================

indices = np.arange(
    len(y_tune)
)

loss_indices, conf_indices = train_test_split(
    indices,
    test_size=0.5,
    stratify=y_tune,
    random_state=42
)

print(
    "\n================ ConfTS INTERNAL SPLIT ================"
)

print(
    "ConfTS loss set:",
    len(loss_indices)
)

print(
    "ConfTS threshold set:",
    len(conf_indices)
)


# ============================================================
# 17. ConfTS GRID SEARCH
# ============================================================

print(
    "\n================ ConfTS GRID SEARCH ================"
)

confts_losses = []

for T in temperatures:

    # --------------------------------------------
    # Convert ALL tuning logits to probabilities
    # --------------------------------------------

    tune_probs = probabilities_from_logits(
        tune_logits,
        T
    )

    tune_scores = aps_score_matrix(
        tune_probs
    )

    # --------------------------------------------
    # Part A:
    # calculate tau(T) from tuning-conf
    # --------------------------------------------

    conf_true_scores = tune_scores[
        conf_indices,
        y_tune[conf_indices]
    ]

    tau_T = conformal_threshold(
        conf_true_scores,
        alpha=0.1
    )

    # --------------------------------------------
    # Part B:
    # true-label scores on tuning-loss
    # --------------------------------------------

    loss_true_scores = tune_scores[
        loss_indices,
        y_tune[loss_indices]
    ]

    loss_scores = tune_scores[loss_indices]

    prediction_sets = loss_scores <= tau_T

    validation_coverage = np.mean(
        prediction_sets[
            np.arange(len(loss_indices)),
            y_tune[loss_indices]
        ]
    )

    validation_avg_size = np.mean(
        prediction_sets.sum(axis=1)
    )

    # --------------------------------------------
    # ConfTS loss:
    #
    #     mean[(tau - S)^2]
    # --------------------------------------------

    confts_loss = np.mean(
        (
            tau_T
            -
            loss_true_scores
        ) ** 2
    )

    confts_losses.append(
        confts_loss
    )

    print(
        f"T = {T:.1f} | "
        f"tau = {tau_T:.6f} | "
        f"ConfTS loss = {confts_loss:.6f} | "
        f"Coverage = {validation_coverage:.4f} | "
        f"Avg size = {validation_avg_size:.4f}"
    )


best_confts_index = np.argmin(
    confts_losses
)

T_ConfTS = temperatures[
    best_confts_index
]

print(
    "\nBest ConfTS-grid temperature:",
    T_ConfTS
)


# ============================================================
# 18. FINAL COMPARISON
# ============================================================

methods = [
    ("Baseline", 1.0),
    ("Temperature Scaling", T_TS),
    ("ConfTS-grid", T_ConfTS)
]

print(
    "\n================ FINAL COMPARISON ================"
)

print(
    f"{'Method':<22}"
    f"{'T':>8}"
    f"{'ECE':>12}"
    f"{'Coverage':>12}"
    f"{'Avg APS Size':>16}"
)

print("-" * 70)


for method_name, T in methods:

    # ECE
    test_probs = probabilities_from_logits(
        test_logits,
        T
    )

    ece = calculate_ece(
        test_probs,
        y_test
    )

    # APS
    coverage, avg_size, tau = evaluate_aps(
        conformal_logits,
        y_conformal,
        test_logits,
        y_test,
        temperature=T,
        alpha=0.1
    )

    print(
        f"{method_name:<22}"
        f"{T:>8.2f}"
        f"{ece:>12.4f}"
        f"{coverage:>12.4f}"
        f"{avg_size:>16.4f}"
    )