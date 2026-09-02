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

ALPHA = 0.1


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

# 2000 conformal calibration, 2000 test
X_conformal, X_test, y_conformal, y_test = train_test_split(
    X_remaining,
    y_remaining,
    test_size=0.5,
    stratify=y_remaining,
    random_state=42
)

print("\n================ SPLITS ================")

print(
    "Training:",
    X_train.shape,
    y_train.shape
)

print(
    "Tuning:",
    X_tune.shape,
    y_tune.shape
)

print(
    "Conformal calibration:",
    X_conformal.shape,
    y_conformal.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)


# ============================================================
# 4. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_tune = scaler.transform(
    X_tune
)

X_conformal = scaler.transform(
    X_conformal
)

X_test = scaler.transform(
    X_test
)


# ============================================================
# 5. PYTORCH TENSORS
# ============================================================

X_train_t = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_t = torch.tensor(
    y_train,
    dtype=torch.long
)

X_tune_t = torch.tensor(
    X_tune,
    dtype=torch.float32
)

y_tune_t = torch.tensor(
    y_tune,
    dtype=torch.long
)

X_conformal_t = torch.tensor(
    X_conformal,
    dtype=torch.float32
)

X_test_t = torch.tensor(
    X_test,
    dtype=torch.float32
)


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

print(
    "\n================ TRAINING ================"
)

for epoch in range(100):

    model.train()

    optimizer.zero_grad()

    logits = model(
        X_train_t
    )

    loss = criterion(
        logits,
        y_train_t
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 20 == 0:

        predictions = torch.argmax(
            logits,
            dim=1
        )

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

    tune_logits = model(
        X_tune_t
    )

    conformal_logits = model(
        X_conformal_t
    )

    test_logits = model(
        X_test_t
    )


# ============================================================
# 9. SOFTMAX WITH TEMPERATURE
# ============================================================

def probabilities_from_logits(
    logits,
    temperature
):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    return (
        probabilities
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# 10. NON-RANDOMIZED APS SCORE MATRIX
# ============================================================

def aps_score_matrix(
    probabilities
):

    # --------------------------------------------
    # Sort class probabilities:
    #
    # biggest → smallest
    # --------------------------------------------

    order = np.argsort(
        -probabilities,
        axis=1
    )

    sorted_probabilities = np.take_along_axis(
        probabilities,
        order,
        axis=1
    )

    # --------------------------------------------
    # Cumulative probabilities
    # --------------------------------------------

    cumulative = np.cumsum(
        sorted_probabilities,
        axis=1
    )

    # --------------------------------------------
    # Put APS scores back into original
    # class positions
    # --------------------------------------------

    scores = np.empty_like(
        cumulative
    )

    rows = np.arange(
        len(probabilities)
    )[:, None]

    scores[
        rows,
        order
    ] = cumulative

    return scores


# ============================================================
# 11. NORMAL CONFORMAL THRESHOLD
# ============================================================

def conformal_threshold(
    true_scores,
    alpha=0.1
):

    n = len(
        true_scores
    )

    k = int(
        np.ceil(
            (n + 1)
            *
            (1 - alpha)
        )
    )

    k = min(
        k,
        n
    )

    sorted_scores = np.sort(
        true_scores
    )

    tau = sorted_scores[
        k - 1
    ]

    return tau


# ============================================================
# 12. APS EVALUATION
# ============================================================

def evaluate_aps(
    conformal_logits,
    y_conformal,
    test_logits,
    y_test,
    temperature,
    alpha=0.1
):

    # ========================================================
    # CONFORMAL CALIBRATION
    # ========================================================

    conformal_probs = probabilities_from_logits(
        conformal_logits,
        temperature
    )

    conformal_scores = aps_score_matrix(
        conformal_probs
    )

    true_conformal_scores = conformal_scores[
        np.arange(
            len(y_conformal)
        ),
        y_conformal
    ]

    tau = conformal_threshold(
        true_conformal_scores,
        alpha
    )

    # ========================================================
    # TEST
    # ========================================================

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
            np.arange(
                len(y_test)
            ),
            y_test
        ]
    )

    average_size = np.mean(
        prediction_sets.sum(
            axis=1
        )
    )

    return (
        coverage,
        average_size,
        tau
    )


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

    for i in range(
        n_bins
    ):

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

        if np.sum(
            in_bin
        ) == 0:

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
# 14. DIFFERENTIABLE APS TRUE-LABEL SCORES
#
# This version stays in PyTorch because we need gradients
# with respect to temperature.
# ============================================================

def aps_true_label_scores_torch(
    logits,
    labels,
    temperature
):

    # --------------------------------------------
    # Temperature-scaled probabilities
    # --------------------------------------------

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    # --------------------------------------------
    # Sort probabilities
    # --------------------------------------------

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        dim=1,
        descending=True
    )

    # --------------------------------------------
    # Cumulative probability = APS scores
    # --------------------------------------------

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    # --------------------------------------------
    # Find where the TRUE class appears
    # in the sorted list
    # --------------------------------------------

    true_positions = (
        sorted_indices
        ==
        labels.unsqueeze(1)
    ).nonzero(
        as_tuple=False
    )[:, 1]

    rows = torch.arange(
        len(labels)
    )

    # --------------------------------------------
    # APS score of true class
    # --------------------------------------------

    true_scores = cumulative_probs[
        rows,
        true_positions
    ]

    return true_scores


# ============================================================
# 15. DIFFERENTIABLE CONFORMAL THRESHOLD
#
# We keep tau as a PyTorch tensor.
# ============================================================

def neural_sort(scores, smoothness=0.1):

    n = scores.shape[-1]

    pairwise_abs_diffs = (
        scores[:, None]
        -
        scores[None, :]
    ).abs()

    pairwise_abs_diffs_sum = (
        pairwise_abs_diffs
        @
        torch.ones(
            n,
            1,
            device=scores.device
        )
    )

    positions = (
        n
        - 1
        - 2
        *
        torch.arange(
            n,
            device=scores.device,
            dtype=scores.dtype
        )
    )

    score_differences = (
        scores[:, None]
        *
        positions
    )

    P_scores = (
        score_differences
        -
        pairwise_abs_diffs_sum
    ).transpose(0, 1)

    P_hat = torch.softmax(
        P_scores / smoothness,
        dim=-1
    )

    return P_hat


def conformal_threshold_torch(
    scores,
    alpha,
    smoothness=0.1
):

    # --------------------------------------------
    # 1. Soft-sort scores
    #
    # NeuralSort gives:
    #
    # largest → smallest
    # --------------------------------------------

    P_hat = neural_sort(
        scores,
        smoothness
    )

    sorted_scores_desc = (
        P_hat
        @
        scores[:, None]
    )[:, 0]

    n = len(scores)

    # --------------------------------------------
    # 2. Normal conformal quantile position
    #
    # For alpha = 0.1:
    #
    # we want approximately the 90th percentile
    # in ASCENDING order.
    # --------------------------------------------

    ascending_index = (
        (1 - alpha)
        *
        (n + 1)
        -
        1
    )

    # --------------------------------------------
    # 3. Convert it because our soft sorting
    # is DESCENDING.
    #
    # Example:
    #
    # ascending index ≈ 900
    #
    # becomes
    #
    # descending index ≈ 99
    # --------------------------------------------

    descending_index = (
        n
        - 1
        - ascending_index
    )

    low = int(
        np.floor(
            descending_index
        )
    )

    high = int(
        np.ceil(
            descending_index
        )
    )

    low = min(
        max(low, 0),
        n - 1
    )

    high = min(
        max(high, 0),
        n - 1
    )

    fraction = (
        descending_index
        -
        np.floor(
            descending_index
        )
    )

    # --------------------------------------------
    # 4. Interpolate between nearby soft scores
    # --------------------------------------------

    tau = (
        sorted_scores_desc[low]
        +
        fraction
        *
        (
            sorted_scores_desc[high]
            -
            sorted_scores_desc[low]
        )
    )

    return tau

# ============================================================
# 16. GRADIENT-BASED ConfTS
# ============================================================

def optimize_confts_gradient(
    tuning_logits,
    tuning_labels,
    conf_indices,
    loss_indices,
    alpha=0.1,
    learning_rate=0.01,
    epochs=200
):

    # --------------------------------------------
    # Convert our existing split indices to Torch
    # --------------------------------------------

    conf_indices_t = torch.tensor(
        conf_indices,
        dtype=torch.long
    )

    loss_indices_t = torch.tensor(
        loss_indices,
        dtype=torch.long
    )

    # --------------------------------------------
    # D_conf
    #
    # Used ONLY to calculate tau(T)
    # --------------------------------------------

    conf_logits = tuning_logits[
        conf_indices_t
    ]

    conf_labels = tuning_labels[
        conf_indices_t
    ]

    # --------------------------------------------
    # D_loss
    #
    # Used to calculate ConfTS loss
    # --------------------------------------------

    loss_logits = tuning_logits[
        loss_indices_t
    ]

    loss_labels = tuning_labels[
        loss_indices_t
    ]

    print(
        "\n================ GRADIENT ConfTS INTERNAL SPLIT ================"
    )

    print(
        "Threshold subset:",
        len(conf_logits)
    )

    print(
        "Loss subset:",
        len(loss_logits)
    )

    # ========================================================
    # THE PARAMETER WE WANT TO LEARN
    #
    # Start at:
    #
    # T = 1.0
    # ========================================================

    temperature = torch.nn.Parameter(
        torch.tensor(
            1.0,
            dtype=torch.float32
        )
    )

    optimizer_T = torch.optim.SGD(
        [temperature],
        lr=learning_rate
    )

    print(
        "\n================ GRADIENT-BASED ConfTS ================"
    )

    for epoch in range(
        epochs
    ):

        optimizer_T.zero_grad()

        # ====================================================
        # STEP A
        #
        # Calculate non-randomized APS scores
        # on D_conf
        # ====================================================

        conf_scores = aps_true_label_scores_torch(
            conf_logits,
            conf_labels,
            temperature
        )

        # ====================================================
        # STEP B
        #
        # Calculate:
        #
        # tau(T)
        # ====================================================

        tau = conformal_threshold_torch(
            conf_scores,
            alpha
        )

        # ====================================================
        # STEP C
        #
        # Calculate true-label APS scores
        # on D_loss
        # ====================================================

        loss_scores = aps_true_label_scores_torch(
            loss_logits,
            loss_labels,
            temperature
        )

        # ====================================================
        # STEP D
        #
        # ConfTS loss:
        #
        #       mean[(tau - S(x,y))²]
        # ====================================================

        loss = torch.mean(
            (
                tau
                -
                loss_scores
            ) ** 2
        )

        # ====================================================
        # STEP E
        #
        # Ask PyTorch:
        #
        # d Loss
        # -------
        #   d T
        # ====================================================

        loss.backward()

        gradient = (
            temperature
            .grad
            .item()
        )

        # ====================================================
        # STEP F
        #
        # Gradient descent:
        #
        # T <- T - learning_rate * gradient
        # ====================================================

        optimizer_T.step()

        # --------------------------------------------
        # Temperature must remain positive.
        #
        # This is just a simple safety guard.
        # --------------------------------------------

        with torch.no_grad():

            temperature.clamp_(
                min=0.05,
                max=5.0
            )

        # --------------------------------------------
        # Print progress
        # --------------------------------------------

        if (
            epoch == 0
            or
            (epoch + 1) % 20 == 0
        ):

            print(
                f"Epoch {epoch + 1:3d} | "
                f"T = {temperature.item():.6f} | "
                f"tau = {tau.item():.6f} | "
                f"loss = {loss.item():.6f} | "
                f"gradient = {gradient:+.6f}"
            )

    return temperature.item()


# ============================================================
# 17. TEMPERATURE CANDIDATES
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
# 18. ORDINARY TEMPERATURE SCALING
#
# Minimize NLL
# ============================================================

print(
    "\n================ ORDINARY TEMPERATURE SCALING ================"
)

nll_results = []

for T in temperatures:

    nll = F.cross_entropy(
        tune_logits / T,
        y_tune_t
    ).item()

    nll_results.append(
        nll
    )

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
# 19. SPLIT TUNING DATA FOR ConfTS
#
# 2000 tuning samples
#
#       ↓
#
# 1000 D_loss
# 1000 D_conf
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
# 20. ConfTS GRID SEARCH
#
# Keep Step 16 so that we can compare:
#
# ConfTS-grid
#
#       vs
#
# ConfTS-gradient
# ============================================================

print(
    "\n================ ConfTS GRID SEARCH ================"
)

confts_losses = []

for T in temperatures:

    # --------------------------------------------
    # Probabilities for all tuning samples
    # --------------------------------------------

    tune_probs = probabilities_from_logits(
        tune_logits,
        T
    )

    # --------------------------------------------
    # APS scores for every possible class
    # --------------------------------------------

    tune_scores = aps_score_matrix(
        tune_probs
    )

    # --------------------------------------------
    # D_conf
    #
    # Calculate tau(T)
    # --------------------------------------------

    conf_true_scores = tune_scores[
        conf_indices,
        y_tune[
            conf_indices
        ]
    ]

    tau_T = conformal_threshold(
        conf_true_scores,
        alpha=ALPHA
    )

    # --------------------------------------------
    # D_loss
    #
    # True-label APS scores
    # --------------------------------------------

    loss_true_scores = tune_scores[
        loss_indices,
        y_tune[
            loss_indices
        ]
    ]

    # --------------------------------------------
    # For educational purposes,
    # also inspect prediction sets on D_loss
    # --------------------------------------------

    all_loss_scores = tune_scores[
        loss_indices
    ]

    prediction_sets = (
        all_loss_scores
        <=
        tau_T
    )

    validation_coverage = np.mean(
        prediction_sets[
            np.arange(
                len(loss_indices)
            ),
            y_tune[
                loss_indices
            ]
        ]
    )

    validation_avg_size = np.mean(
        prediction_sets.sum(
            axis=1
        )
    )

    # --------------------------------------------
    # ConfTS loss
    #
    # mean[(tau - true APS score)^2]
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

T_ConfTS_grid = temperatures[
    best_confts_index
]

print(
    "\nBest ConfTS-grid temperature:",
    T_ConfTS_grid
)


# ============================================================
# 21. GRADIENT-BASED ConfTS
#
# THIS IS THE NEW PART OF STEP 17
# ============================================================

T_ConfTS_gradient = optimize_confts_gradient(

    tuning_logits=tune_logits,

    tuning_labels=y_tune_t,

    conf_indices=conf_indices,

    loss_indices=loss_indices,

    alpha=ALPHA,

    learning_rate=0.01,

    epochs=200
)


print(
    "\nGradient ConfTS temperature:",
    T_ConfTS_gradient
)


# ============================================================
# 22. COMPARE THE TWO ConfTS TEMPERATURES
# ============================================================

print(
    "\n================ GRID vs GRADIENT ================"
)

print(
    f"ConfTS-grid temperature:     "
    f"{T_ConfTS_grid:.6f}"
)

print(
    f"ConfTS-gradient temperature: "
    f"{T_ConfTS_gradient:.6f}"
)

print(
    f"Difference:                  "
    f"{abs(T_ConfTS_grid - T_ConfTS_gradient):.6f}"
)


# ============================================================
# 23. FINAL COMPARISON
#
# IMPORTANT:
#
# Now we use the completely separate
# conformal calibration set.
#
# The tuning subsets are finished.
# ============================================================

methods = [

    (
        "Baseline",
        1.0
    ),

    (
        "Temperature Scaling",
        float(T_TS)
    ),

    (
        "ConfTS-grid",
        float(T_ConfTS_grid)
    ),

    (
        "ConfTS-gradient",
        float(T_ConfTS_gradient)
    )
]


print(
    "\n================ FINAL COMPARISON ================"
)

print(
    f"{'Method':<22}"
    f"{'T':>10}"
    f"{'ECE':>12}"
    f"{'Coverage':>12}"
    f"{'Avg APS Size':>16}"
    f"{'Tau':>12}"
)

print(
    "-" * 84
)


for method_name, T in methods:

    # ========================================================
    # ECE ON TEST DATA
    # ========================================================

    test_probs = probabilities_from_logits(
        test_logits,
        T
    )

    ece = calculate_ece(
        test_probs,
        y_test
    )

    # ========================================================
    # APS
    #
    # tau is calculated using the SEPARATE
    # conformal calibration set.
    # ========================================================

    coverage, avg_size, tau = evaluate_aps(

        conformal_logits,
        y_conformal,

        test_logits,
        y_test,

        temperature=T,

        alpha=ALPHA
    )

    print(
        f"{method_name:<22}"
        f"{T:>10.4f}"
        f"{ece:>12.4f}"
        f"{coverage:>12.4f}"
        f"{avg_size:>16.4f}"
        f"{tau:>12.6f}"
    )