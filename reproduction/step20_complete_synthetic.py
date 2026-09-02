import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 20 — COMPLETE SYNTHETIC REPRODUCTION
#
# IMPORTANT:
# - ConfTS optimization still uses NON-RANDOMIZED APS.
# - Final conformal calibration + prediction sets use randomized APS/RAPS.
# - Final comparison: Baseline vs Temperature Scaling vs ConfTS.
# ============================================================


# ============================================================
# 1. Reproducibility
# ============================================================

np.random.seed(42)
torch.manual_seed(42)


# ============================================================
# 2. Create the SAME 20-class synthetic dataset
#    used in Steps 12–17
# ============================================================

X, y = make_classification(
    n_samples=12000,
    n_features=20,
    n_informative=18,
    n_redundant=2,
    n_classes=20,
    n_clusters_per_class=1,
    class_sep=2.0,
    flip_y=0.01,
    random_state=42
)

print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3. Create the SAME four main splits
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
# 4. Standardize features
#
# Fit ONLY on training data.
# Then apply the same transformation to all other splits.
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_tuning = scaler.transform(X_tuning)
X_conformal = scaler.transform(X_conformal)
X_test = scaler.transform(X_test)


# ============================================================
# 5. Convert training data to PyTorch tensors
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
# 6. SAME tiny neural network as Steps 12–17
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
# 7. Train the classifier
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 100

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
# 8. Get logits for tuning, conformal, and test data
# ============================================================

model.eval()

X_tuning_tensor = torch.tensor(
    X_tuning,
    dtype=torch.float32
)

X_conformal_tensor = torch.tensor(
    X_conformal,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

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
# 9. General settings
# ============================================================

alpha = 0.10

# RAPS settings learned in Step 19
k_reg = 1
lambda_reg = 0.001

temperatures = np.round(
    np.arange(0.1, 1.31, 0.1),
    1
)


# ============================================================
# 10. Helper: logits -> probabilities
# ============================================================

def probabilities_from_logits(logits, temperature):

    with torch.no_grad():

        probabilities = torch.softmax(
            logits / float(temperature),
            dim=1
        )

    return probabilities.numpy()


# ============================================================
# 11. NON-RANDOMIZED APS score matrix
#
# This is still used for:
# - ConfTS-grid optimization
# - ConfTS-gradient optimization
#
# For each sample and candidate class:
# score = cumulative probability INCLUDING that class
# ============================================================

def aps_score_matrix(probabilities):

    order = np.argsort(
        -probabilities,
        axis=1
    )

    sorted_probs = np.take_along_axis(
        probabilities,
        order,
        axis=1
    )

    cumulative_probs = np.cumsum(
        sorted_probs,
        axis=1
    )

    scores = np.empty_like(
        probabilities
    )

    np.put_along_axis(
        scores,
        order,
        cumulative_probs,
        axis=1
    )

    return scores


# ============================================================
# 12. NEW IN STEP 18:
#     RANDOMIZED APS score matrix
#
# For ONE sample, one random u is used for all candidate labels.
#
# score(class k)
# = probability mass BEFORE class k
#   + u * probability(class k)
# ============================================================

def randomized_aps_score_matrix(probabilities, u_values):

    order = np.argsort(
        -probabilities,
        axis=1
    )

    sorted_probs = np.take_along_axis(
        probabilities,
        order,
        axis=1
    )

    cumulative_probs = np.cumsum(
        sorted_probs,
        axis=1
    )

    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )

    randomized_sorted_scores = (
        cumulative_before
        + u_values[:, None] * sorted_probs
    )

    scores = np.empty_like(
        probabilities
    )

    np.put_along_axis(
        scores,
        order,
        randomized_sorted_scores,
        axis=1
    )

    return scores


# ============================================================
# 13. RANDOMIZED RAPS score matrix
#
# RAPS = randomized APS
#        + lambda * max(rank - k_reg, 0)
# ============================================================

def randomized_raps_score_matrix(
    probabilities,
    u_values,
    k_reg=1,
    lambda_reg=0.001
):

    order = np.argsort(
        -probabilities,
        axis=1
    )

    sorted_probs = np.take_along_axis(
        probabilities,
        order,
        axis=1
    )

    cumulative_probs = np.cumsum(
        sorted_probs,
        axis=1
    )

    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )

    # Human ranks are 1, 2, ..., K.
    ranks = np.arange(
        1,
        probabilities.shape[1] + 1
    )

    penalties = (
        lambda_reg
        * np.maximum(
            ranks - k_reg,
            0
        )
    )

    randomized_sorted_scores = (
        cumulative_before
        + u_values[:, None] * sorted_probs
        + penalties[None, :]
    )

    scores = np.empty_like(
        probabilities
    )

    np.put_along_axis(
        scores,
        order,
        randomized_sorted_scores,
        axis=1
    )

    return scores


# ============================================================
# 14. Standard conformal threshold
# ============================================================

def conformal_threshold(calibration_scores, alpha):

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

    sorted_scores = np.sort(
        calibration_scores
    )

    tau = sorted_scores[
        rank - 1
    ]

    return tau


# ============================================================
# 14. ECE
# ============================================================

def calculate_ece(probabilities, true_labels, n_bins=10):

    confidences = np.max(
        probabilities,
        axis=1
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    correct = (
        predictions == true_labels
    )

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            in_bin = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            in_bin = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if np.any(in_bin):

            bin_accuracy = np.mean(
                correct[in_bin]
            )

            bin_confidence = np.mean(
                confidences[in_bin]
            )

            bin_weight = np.mean(
                in_bin
            )

            ece += (
                bin_weight
                * abs(
                    bin_accuracy
                    - bin_confidence
                )
            )

    return ece


# ============================================================
# 15. NON-RANDOMIZED APS evaluator
#
# Kept only so we can compare Step 17 vs Step 18.
# ============================================================

def evaluate_nonrandomized_aps(temperature):

    conformal_probs = probabilities_from_logits(
        conformal_logits,
        temperature
    )

    test_probs = probabilities_from_logits(
        test_logits,
        temperature
    )

    conformal_scores_all = aps_score_matrix(
        conformal_probs
    )

    calibration_scores = conformal_scores_all[
        np.arange(len(y_conformal)),
        y_conformal
    ]

    tau = conformal_threshold(
        calibration_scores,
        alpha
    )

    test_scores_all = aps_score_matrix(
        test_probs
    )

    included = (
        test_scores_all <= tau
    )

    coverage = np.mean(
        included[
            np.arange(len(y_test)),
            y_test
        ]
    )

    average_size = np.mean(
        np.sum(
            included,
            axis=1
        )
    )

    empty_sets = np.sum(
        np.sum(
            included,
            axis=1
        ) == 0
    )

    return (
        tau,
        coverage,
        average_size,
        int(empty_sets)
    )


# ============================================================
# 16. NEW: RANDOMIZED APS evaluator
#
# IMPORTANT:
# We use randomization BOTH when calculating tau
# and when generating the final test prediction sets.
# ============================================================

def evaluate_randomized_aps(
    temperature,
    u_conformal,
    u_test
):

    conformal_probs = probabilities_from_logits(
        conformal_logits,
        temperature
    )

    test_probs = probabilities_from_logits(
        test_logits,
        temperature
    )

    # ------------------------------
    # Randomized conformal scores
    # ------------------------------

    conformal_scores_all = randomized_aps_score_matrix(
        conformal_probs,
        u_conformal
    )

    calibration_scores = conformal_scores_all[
        np.arange(len(y_conformal)),
        y_conformal
    ]

    tau = conformal_threshold(
        calibration_scores,
        alpha
    )

    # ------------------------------
    # Randomized test scores
    # ------------------------------

    test_scores_all = randomized_aps_score_matrix(
        test_probs,
        u_test
    )

    included = (
        test_scores_all <= tau
    )

    coverage = np.mean(
        included[
            np.arange(len(y_test)),
            y_test
        ]
    )

    set_sizes = np.sum(
        included,
        axis=1
    )

    average_size = np.mean(
        set_sizes
    )

    empty_sets = np.sum(
        set_sizes == 0
    )

    return (
        tau,
        coverage,
        average_size,
        int(empty_sets)
    )


# ============================================================
# 17. RANDOMIZED RAPS evaluator
# ============================================================

def evaluate_randomized_raps(
    temperature,
    u_conformal,
    u_test,
    k_reg=1,
    lambda_reg=0.001
):

    conformal_probs = probabilities_from_logits(
        conformal_logits,
        temperature
    )

    test_probs = probabilities_from_logits(
        test_logits,
        temperature
    )

    conformal_scores_all = randomized_raps_score_matrix(
        conformal_probs,
        u_conformal,
        k_reg=k_reg,
        lambda_reg=lambda_reg
    )

    calibration_scores = conformal_scores_all[
        np.arange(len(y_conformal)),
        y_conformal
    ]

    tau = conformal_threshold(
        calibration_scores,
        alpha
    )

    test_scores_all = randomized_raps_score_matrix(
        test_probs,
        u_test,
        k_reg=k_reg,
        lambda_reg=lambda_reg
    )

    included = (
        test_scores_all <= tau
    )

    coverage = np.mean(
        included[
            np.arange(len(y_test)),
            y_test
        ]
    )

    set_sizes = np.sum(
        included,
        axis=1
    )

    average_size = np.mean(
        set_sizes
    )

    empty_sets = np.sum(
        set_sizes == 0
    )

    return (
        tau,
        coverage,
        average_size,
        int(empty_sets)
    )


# ============================================================
# 18. Ordinary Temperature Scaling
#     choose T using minimum NLL on the tuning set
# ============================================================

y_tuning_tensor = torch.tensor(
    y_tuning,
    dtype=torch.long
)

best_ts_temperature = None
best_ts_nll = float("inf")

for T in temperatures:

    with torch.no_grad():

        nll = nn.functional.cross_entropy(
            tuning_logits / float(T),
            y_tuning_tensor
        ).item()

    if nll < best_ts_nll:

        best_ts_nll = nll
        best_ts_temperature = float(T)


print("\n================ ORDINARY TEMPERATURE SCALING ================")
print(f"Best T by NLL: {best_ts_temperature:.4f}")
print(f"Best tuning NLL: {best_ts_nll:.6f}")


# ============================================================
# 18. Split the TUNING set for ConfTS
#
# D_loss = 1000 samples
# D_conf = 1000 samples
#
# IMPORTANT:
# This is separate from the final 2000-sample conformal set.
# ============================================================

(
    X_tune_loss,
    X_tune_conf,
    y_tune_loss,
    y_tune_conf
) = train_test_split(
    X_tuning,
    y_tuning,
    test_size=1000,
    random_state=42,
    stratify=y_tuning
)

X_tune_loss_tensor = torch.tensor(
    X_tune_loss,
    dtype=torch.float32
)

X_tune_conf_tensor = torch.tensor(
    X_tune_conf,
    dtype=torch.float32
)

y_tune_loss_tensor = torch.tensor(
    y_tune_loss,
    dtype=torch.long
)

y_tune_conf_tensor = torch.tensor(
    y_tune_conf,
    dtype=torch.long
)

with torch.no_grad():

    tune_loss_logits = model(
        X_tune_loss_tensor
    )

    tune_conf_logits = model(
        X_tune_conf_tensor
    )

print("\n================ CONFTS TUNING SPLIT ================")
print("D_loss:", X_tune_loss.shape, y_tune_loss.shape)
print("D_conf:", X_tune_conf.shape, y_tune_conf.shape)


# ============================================================
# 19. ConfTS-grid from Step 16
#
# IMPORTANT:
# Uses NON-RANDOMIZED APS.
# ============================================================

def confts_grid_loss(temperature):

    conf_probs = probabilities_from_logits(
        tune_conf_logits,
        temperature
    )

    loss_probs = probabilities_from_logits(
        tune_loss_logits,
        temperature
    )

    conf_scores_all = aps_score_matrix(
        conf_probs
    )

    conf_true_scores = conf_scores_all[
        np.arange(len(y_tune_conf)),
        y_tune_conf
    ]

    tau = conformal_threshold(
        conf_true_scores,
        alpha
    )

    loss_scores_all = aps_score_matrix(
        loss_probs
    )

    loss_true_scores = loss_scores_all[
        np.arange(len(y_tune_loss)),
        y_tune_loss
    ]

    loss = np.mean(
        (
            tau
            - loss_true_scores
        ) ** 2
    )

    return tau, loss


best_grid_temperature = None
best_grid_loss = float("inf")

print("\n================ CONFTS GRID SEARCH ================")
print("T      Tau          ConfTS Loss")
print("-----------------------------------")

for T in temperatures:

    tau_grid, grid_loss = confts_grid_loss(
        float(T)
    )

    print(
        f"{T:<4.1f}   "
        f"{tau_grid:<12.8f} "
        f"{grid_loss:.8f}"
    )

    if grid_loss < best_grid_loss:

        best_grid_loss = grid_loss
        best_grid_temperature = float(T)

print("\nBest ConfTS-grid T:", best_grid_temperature)
print("Best ConfTS-grid loss:", best_grid_loss)


# ============================================================
# 20. Differentiable NON-RANDOMIZED true-label APS scores
#     used by gradient ConfTS
# ============================================================

def aps_true_label_scores_torch(
    logits,
    labels,
    temperature
):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        dim=1,
        descending=True
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    true_positions = (
        sorted_indices
        == labels.unsqueeze(1)
    ).float().argmax(
        dim=1
    )

    true_scores = cumulative_probs.gather(
        1,
        true_positions.unsqueeze(1)
    ).squeeze(1)

    return true_scores


# ============================================================
# 21. NeuralSort
#
# We need a SOFT/differentiable approximation to sorting
# so gradient can flow through the conformal threshold.
# ============================================================

def neural_sort(scores, smoothness=0.1):

    n = scores.shape[0]

    scores_column = scores.view(
        -1,
        1
    )

    pairwise_absolute_difference = torch.abs(
        scores_column
        - scores_column.T
    )

    row_sum = pairwise_absolute_difference.sum(
        dim=1,
        keepdim=True
    )

    positions = torch.arange(
        1,
        n + 1,
        device=scores.device,
        dtype=scores.dtype
    )

    scaling = (
        n + 1
        - 2 * positions
    ).view(
        1,
        -1
    )

    score_position_product = (
        scores_column
        @ scaling
    )

    logits = (
        score_position_product
        - row_sum
    ).T

    soft_permutation = torch.softmax(
        logits / smoothness,
        dim=1
    )

    softly_sorted_scores = (
        soft_permutation
        @ scores
    )

    # Descending order
    return softly_sorted_scores


# ============================================================
# 22. Differentiable conformal threshold
#
# NeuralSort gives descending scores.
# We convert the ordinary ascending conformal-quantile position
# into the corresponding descending position.
# ============================================================

def conformal_threshold_torch(
    scores,
    alpha,
    smoothness=0.1
):

    n = scores.shape[0]

    sorted_descending = neural_sort(
        scores,
        smoothness=smoothness
    )

    # Continuous version of the conformal rank position.
    ascending_index = (
        (1 - alpha)
        * (n + 1)
        - 1
    )

    ascending_index = max(
        0.0,
        min(
            float(n - 1),
            float(ascending_index)
        )
    )

    descending_index = (
        (n - 1)
        - ascending_index
    )

    lower_index = int(
        math.floor(
            descending_index
        )
    )

    upper_index = int(
        math.ceil(
            descending_index
        )
    )

    if lower_index == upper_index:

        return sorted_descending[
            lower_index
        ]

    interpolation_weight = (
        descending_index
        - lower_index
    )

    tau = (
        (1 - interpolation_weight)
        * sorted_descending[lower_index]
        + interpolation_weight
        * sorted_descending[upper_index]
    )

    return tau


# ============================================================
# 23. Sanity check: hard tau vs soft tau at T = 1
# ============================================================

with torch.no_grad():

    scores_at_1 = aps_true_label_scores_torch(
        tune_conf_logits,
        y_tune_conf_tensor,
        torch.tensor(1.0)
    )

hard_tau_at_1 = conformal_threshold(
    scores_at_1.numpy(),
    alpha
)

soft_tau_at_1 = conformal_threshold_torch(
    scores_at_1,
    alpha,
    smoothness=0.1
).item()

print("\n================ SOFT THRESHOLD CHECK ================")
print(f"Hard tau at T=1.0: {hard_tau_at_1:.6f}")
print(f"Soft tau at T=1.0: {soft_tau_at_1:.6f}")


# ============================================================
# 24. Gradient-based ConfTS from Step 17
#
# IMPORTANT:
# STILL NON-RANDOMIZED APS.
# ============================================================

def optimize_confts_gradient(
    initial_temperature=1.0,
    learning_rate=0.01,
    epochs=1000
):

    temperature = torch.nn.Parameter(
        torch.tensor(
            float(initial_temperature),
            dtype=torch.float32
        )
    )

    optimizer_temperature = torch.optim.SGD(
        [temperature],
        lr=learning_rate
    )

    first_loss = None

    print("\n================ CONFTS GRADIENT ================")

    for epoch in range(1, epochs + 1):

        optimizer_temperature.zero_grad()

        conf_scores = aps_true_label_scores_torch(
            tune_conf_logits,
            y_tune_conf_tensor,
            temperature
        )

        tau = conformal_threshold_torch(
            conf_scores,
            alpha,
            smoothness=0.1
        )

        loss_scores = aps_true_label_scores_torch(
            tune_loss_logits,
            y_tune_loss_tensor,
            temperature
        )

        confts_loss = torch.mean(
            (
                tau
                - loss_scores
            ) ** 2
        )

        if first_loss is None:
            first_loss = confts_loss.item()

        confts_loss.backward()

        optimizer_temperature.step()

        # Keep T positive and away from numerical disaster.
        with torch.no_grad():
            temperature.clamp_(
                0.05,
                5.0
            )

        if (
            epoch == 1
            or epoch % 20 == 0
        ):

            print(
                f"Epoch {epoch:3d} | "
                f"T: {temperature.item():.6f} | "
                f"Tau: {tau.item():.6f} | "
                f"Loss: {confts_loss.item():.6f}"
            )

    return (
        temperature.item(),
        first_loss,
        confts_loss.item()
    )


(
    gradient_temperature,
    gradient_start_loss,
    gradient_final_loss
) = optimize_confts_gradient()

print("\nGradient ConfTS summary:")
print(
    f"T: 1.000000 -> "
    f"{gradient_temperature:.6f}"
)
print(
    f"Loss: {gradient_start_loss:.6f} -> "
    f"{gradient_final_loss:.6f}"
)


# ============================================================
# 25. Tiny randomized APS sanity check
#     This is the exact example from our discussion.
# ============================================================

example_probs = np.array([
    [0.60, 0.20, 0.10, 0.07, 0.03]
])

example_u = np.array([
    0.50
])

example_scores = randomized_aps_score_matrix(
    example_probs,
    example_u
)[0]

print("\n================ RANDOMIZED APS EXAMPLE ================")
print("Probabilities:", example_probs[0])
print("u:", example_u[0])
print(
    "Randomized APS scores:",
    np.round(
        example_scores,
        3
    )
)


# ============================================================
# 26. Generate the random u values for STANDARD APS
#
# One u per conformal sample.
# One u per test sample.
#
# We keep these SAME u values for every method so the
# Baseline / TS / ConfTS comparison is fair.
# ============================================================

rng = np.random.default_rng(42)

u_conformal = rng.uniform(
    0.0,
    1.0,
    size=len(y_conformal)
)

u_test = rng.uniform(
    0.0,
    1.0,
    size=len(y_test)
)

print("\n================ RANDOM U VALUES ================")
print(
    "First 5 conformal u:",
    np.round(
        u_conformal[:5],
        4
    )
)
print(
    "First 5 test u:",
    np.round(
        u_test[:5],
        4
    )
)


# ============================================================
# 27. Compare Step 17 non-randomized APS
#     with Step 18 randomized APS at T = 1
# ============================================================

(
    old_tau,
    old_coverage,
    old_size,
    old_empty
) = evaluate_nonrandomized_aps(
    1.0
)

(
    new_tau,
    new_coverage,
    new_size,
    new_empty
) = evaluate_randomized_aps(
    1.0,
    u_conformal,
    u_test
)

print("\n================ NON-RANDOMIZED VS RANDOMIZED APS ================")
print("Method             Tau        Coverage    Avg Size    Empty")
print("-----------------------------------------------------------------")
print(
    f"Non-randomized     "
    f"{old_tau:<10.6f} "
    f"{old_coverage:<11.4f} "
    f"{old_size:<11.4f} "
    f"{old_empty}"
)
print(
    f"Randomized         "
    f"{new_tau:<10.6f} "
    f"{new_coverage:<11.4f} "
    f"{new_size:<11.4f} "
    f"{new_empty}"
)


# ============================================================
# 28. FINAL STEP-20 COMPARISON
#
# SAME trained model.
# SAME test set.
# SAME random u values.
# Only the temperature-selection method changes.
# ============================================================

methods = [
    (
        "Baseline",
        1.0
    ),
    (
        "Temperature Scaling",
        best_ts_temperature
    ),
    (
        "ConfTS",
        gradient_temperature
    )
]

print("\n================ FINAL STEP 20 RESULTS ================")
print(
    f"{'Method':<22}"
    f"{'T':>9}"
    f"{'ECE':>10}"
    f"{'APS Cov':>11}"
    f"{'APS Size':>11}"
    f"{'RAPS Cov':>12}"
    f"{'RAPS Size':>12}"
)
print("-" * 87)

for method_name, T in methods:

    # ECE depends only on the temperature-scaled probabilities.
    test_probs = probabilities_from_logits(
        test_logits,
        T
    )

    ece = calculate_ece(
        test_probs,
        y_test
    )

    (
        aps_tau,
        aps_coverage,
        aps_average_size,
        aps_empty
    ) = evaluate_randomized_aps(
        T,
        u_conformal,
        u_test
    )

    (
        raps_tau,
        raps_coverage,
        raps_average_size,
        raps_empty
    ) = evaluate_randomized_raps(
        T,
        u_conformal,
        u_test,
        k_reg=k_reg,
        lambda_reg=lambda_reg
    )

    print(
        f"{method_name:<22}"
        f"{T:>9.4f}"
        f"{ece:>10.4f}"
        f"{aps_coverage:>11.4f}"
        f"{aps_average_size:>11.4f}"
        f"{raps_coverage:>12.4f}"
        f"{raps_average_size:>12.4f}"
    )


# ============================================================
# 29. FINAL REMINDER
# ============================================================

print("\n================ STEP 20 REMINDER ================")
print("Baseline temperature: T = 1.0")
print("Ordinary TS: choose T by minimum tuning NLL")
print("ConfTS: choose T by non-randomized APS efficiency-gap loss")
print("Final APS/RAPS: randomized conformal prediction")
print(f"RAPS settings: k_reg={k_reg}, lambda={lambda_reg}")
