import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. Load the cached logits
# ============================================================

cache = torch.load(
    "cifar100_logits_cache.pt",
    map_location="cpu"
)

tuning_logits = cache["tuning_logits"]
tuning_labels = cache["tuning_labels"]

conformal_logits = cache["conformal_logits"]
conformal_labels = cache["conformal_labels"]

test_logits = cache["test_logits"]
test_labels = cache["test_labels"]


print("================ CACHED DATA ================")

print("Tuning:", tuning_logits.shape)
print("Conformal:", conformal_logits.shape)
print("Test:", test_logits.shape)


# ============================================================
# 2. Temperature-scaled probabilities
# ============================================================

def probabilities_from_logits(logits, temperature):

    return torch.softmax(
        logits / temperature,
        dim=1
    )


# ============================================================
# 3. Expected Calibration Error
# ============================================================

def calculate_ece(probabilities, labels, n_bins=15):

    confidences, predictions = probabilities.max(dim=1)

    correct = (
        predictions == labels
    ).float()

    ece = 0.0

    bin_boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        in_bin = (
            (confidences > lower)
            &
            (confidences <= upper)
        )

        if in_bin.sum() > 0:

            bin_accuracy = correct[in_bin].mean()

            bin_confidence = confidences[in_bin].mean()

            bin_fraction = in_bin.float().mean()

            ece += (
                torch.abs(
                    bin_accuracy
                    - bin_confidence
                )
                * bin_fraction
            )

    return ece.item()


# ============================================================
# 4. Baseline
#
# T = 1 means:
#
# logits / 1 = original logits
# ============================================================

baseline_T = 1.0

baseline_test_probs = probabilities_from_logits(
    test_logits,
    baseline_T
)

baseline_ece = calculate_ece(
    baseline_test_probs,
    test_labels
)


print("\n================ BASELINE ================")

print(f"T:   {baseline_T:.3f}")
print(f"ECE: {baseline_ece:.4f}")


# ============================================================
# 5. Ordinary Temperature Scaling
#
# Try many possible temperatures.
#
# For each T:
#
# logits / T
#       ↓
# cross-entropy / NLL
#
# Choose the T with the LOWEST tuning NLL.
# ============================================================

temperatures = torch.arange(
    0.2,
    3.01,
    0.05
)


best_T = None
best_nll = float("inf")


print("\n================ TS GRID SEARCH ================")

for T_tensor in temperatures:

    T = T_tensor.item()

    scaled_logits = (
        tuning_logits / T
    )

    nll = F.cross_entropy(
        scaled_logits,
        tuning_labels
    ).item()

    if nll < best_nll:

        best_nll = nll
        best_T = T


print(
    f"Best TS temperature: {best_T:.3f}"
)

print(
    f"Best tuning NLL:     {best_nll:.4f}"
)


# ============================================================
# 6. Evaluate TS on FINAL TEST data
# ============================================================

ts_test_probs = probabilities_from_logits(
    test_logits,
    best_T
)

ts_ece = calculate_ece(
    ts_test_probs,
    test_labels
)


print("\n================ BASELINE vs TS ================")

print(
    f"{'Method':<12}"
    f"{'T':>10}"
    f"{'ECE':>12}"
)

print(
    f"{'Baseline':<12}"
    f"{baseline_T:>10.3f}"
    f"{baseline_ece:>12.4f}"
)

print(
    f"{'TS':<12}"
    f"{best_T:>10.3f}"
    f"{ts_ece:>12.4f}"
)


# ============================================================
# 7. Randomized APS
# ============================================================

alpha = 0.10


# ------------------------------------------------------------
# Use the SAME random numbers for Baseline and TS.
#
# This makes the comparison fair:
#
# differences should come from temperature,
# not from different random u values.
# ------------------------------------------------------------

torch.manual_seed(42)

u_conformal = torch.rand(
    len(conformal_labels)
)

u_test = torch.rand(
    len(test_labels),
    100
)


# ============================================================
# 8. APS calibration scores
# ============================================================

def aps_calibration_scores(
    logits,
    labels,
    temperature,
    u
):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    # Sort probabilities:
    #
    # biggest → smallest
    #
    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        dim=1,
        descending=True
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    # APS randomized score:
    #
    # probabilities BEFORE true class
    # +
    # u * probability of true class
    #
    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )

    # Find the rank of the true class.
    #
    # Example:
    #
    # sorted labels = [7, 2, 5, 1, ...]
    # true label = 5
    #
    # true rank = 2
    #
    true_positions = (
        sorted_indices
        == labels.unsqueeze(1)
    ).nonzero(as_tuple=False)[:, 1]

    rows = torch.arange(
        len(labels)
    )

    true_probs = sorted_probs[
        rows,
        true_positions
    ]

    before_true = cumulative_before[
        rows,
        true_positions
    ]

    scores = (
        before_true
        + u * true_probs
    )

    return scores


# ============================================================
# 9. Exact conformal threshold
# ============================================================

def conformal_threshold(
    scores,
    alpha
):

    n = len(scores)

    k = int(
        torch.ceil(
            torch.tensor(
                (n + 1) * (1 - alpha)
            )
        ).item()
    )

    k = min(k, n)

    tau = torch.kthvalue(
        scores,
        k
    ).values.item()

    return tau


# ============================================================
# 10. Generate APS test prediction sets
# ============================================================

def evaluate_aps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    temperature,
    alpha,
    u_conformal,
    u_test
):

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    calibration_scores = aps_calibration_scores(
        conformal_logits,
        conformal_labels,
        temperature,
        u_conformal
    )

    tau = conformal_threshold(
        calibration_scores,
        alpha
    )


    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        test_logits / temperature,
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

    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )


    # --------------------------------------------------------
    # Randomized score for EVERY candidate class
    # --------------------------------------------------------

    sorted_scores = (
        cumulative_before
        + u_test * sorted_probs
    )


    # A class enters the prediction set when:
    #
    # APS score <= tau
    #
    included_sorted = (
        sorted_scores <= tau
    )


    # --------------------------------------------------------
    # Average prediction-set size
    # --------------------------------------------------------

    set_sizes = included_sorted.sum(
        dim=1
    )

    average_size = (
        set_sizes.float()
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Coverage
    #
    # Convert membership back to original class order.
    # --------------------------------------------------------

    included_original = torch.zeros_like(
        included_sorted
    )

    included_original.scatter_(
        1,
        sorted_indices,
        included_sorted
    )

    rows = torch.arange(
        len(test_labels)
    )

    covered = included_original[
        rows,
        test_labels
    ]

    coverage = (
        covered.float()
        .mean()
        .item()
    )


    return tau, coverage, average_size


# ============================================================
# 11. Baseline APS
# ============================================================

baseline_tau, baseline_coverage, baseline_aps_size = evaluate_aps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    baseline_T,
    alpha,
    u_conformal,
    u_test
)


# ============================================================
# 12. Temperature Scaling APS
# ============================================================

ts_tau, ts_coverage, ts_aps_size = evaluate_aps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    best_T,
    alpha,
    u_conformal,
    u_test
)


# ============================================================
# 13. Final comparison so far
# ============================================================

print("\n================ CIFAR-100 APS ================")

print(
    f"{'Method':<12}"
    f"{'T':>8}"
    f"{'ECE':>10}"
    f"{'Tau':>10}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)

print(
    f"{'Baseline':<12}"
    f"{baseline_T:>8.3f}"
    f"{baseline_ece:>10.4f}"
    f"{baseline_tau:>10.4f}"
    f"{baseline_coverage:>12.4f}"
    f"{baseline_aps_size:>12.4f}"
)

print(
    f"{'TS':<12}"
    f"{best_T:>8.3f}"
    f"{ts_ece:>10.4f}"
    f"{ts_tau:>10.4f}"
    f"{ts_coverage:>12.4f}"
    f"{ts_aps_size:>12.4f}"
)

# ============================================================
# 14. ConfTS tuning split
#
# The 2000 tuning examples are split into:
#
# 1000 → calculate tau(T)
# 1000 → calculate ConfTS loss
# ============================================================

from sklearn.model_selection import train_test_split


all_tuning_indices = list(
    range(len(tuning_labels))
)


tau_indices, loss_indices = train_test_split(
    all_tuning_indices,
    test_size=1000,
    random_state=42,
    stratify=tuning_labels.numpy()
)


tau_indices = torch.tensor(
    tau_indices,
    dtype=torch.long
)

loss_indices = torch.tensor(
    loss_indices,
    dtype=torch.long
)


confTS_tau_logits = tuning_logits[
    tau_indices
]

confTS_tau_labels = tuning_labels[
    tau_indices
]


confTS_loss_logits = tuning_logits[
    loss_indices
]

confTS_loss_labels = tuning_labels[
    loss_indices
]


print("\n================ ConfTS SPLIT ================")

print(
    "Tau subset:",
    confTS_tau_logits.shape
)

print(
    "Loss subset:",
    confTS_loss_logits.shape
)


# ============================================================
# 15. Non-randomized APS true-label scores
#
# Example:
#
# probabilities:
#
# A = 0.60
# B = 0.20
# C = 0.10
#
# If true class = C:
#
# score = 0.60 + 0.20 + 0.10
#       = 0.90
#
# No random u here.
# ============================================================

def nonrandomized_aps_scores(
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
    ).nonzero(as_tuple=False)[:, 1]


    rows = torch.arange(
        len(labels)
    )


    scores = cumulative_probs[
        rows,
        true_positions
    ]


    return scores


# ============================================================
# 16. ConfTS grid search
# ============================================================

confTS_temperatures = torch.arange(
    0.30,
    1.51,
    0.025
)


best_confTS_T = None
best_confTS_loss = float("inf")
best_confTS_tau = None


print("\n================ ConfTS GRID SEARCH ================")


for T_tensor in confTS_temperatures:

    T = T_tensor.item()


    # --------------------------------------------------------
    # First subset:
    # calculate non-randomized APS scores
    # and obtain tau(T)
    # --------------------------------------------------------

    tau_scores = nonrandomized_aps_scores(
        confTS_tau_logits,
        confTS_tau_labels,
        T
    )


    tau_T = conformal_threshold(
        tau_scores,
        alpha
    )


    # --------------------------------------------------------
    # Second subset:
    # calculate true-label scores
    # --------------------------------------------------------

    loss_scores = nonrandomized_aps_scores(
        confTS_loss_logits,
        confTS_loss_labels,
        T
    )


    # --------------------------------------------------------
    # ConfTS loss:
    #
    # mean[(tau(T) - S(x,y,T))²]
    # --------------------------------------------------------

    confTS_loss = torch.mean(
        (
            tau_T
            - loss_scores
        ) ** 2
    ).item()


    if confTS_loss < best_confTS_loss:

        best_confTS_loss = confTS_loss
        best_confTS_T = T
        best_confTS_tau = tau_T


print(
    f"Best ConfTS temperature: {best_confTS_T:.3f}"
)

print(
    f"Best ConfTS loss:        {best_confTS_loss:.6f}"
)

print(
    f"ConfTS tuning tau:       {best_confTS_tau:.4f}"
)


# ============================================================
# 17. ConfTS ECE on final test set
#
# Remember:
# ConfTS is NOT trying to minimize ECE.
# ============================================================

confTS_test_probs = probabilities_from_logits(
    test_logits,
    best_confTS_T
)


confTS_ece = calculate_ece(
    confTS_test_probs,
    test_labels
)


# ============================================================
# 18. Final APS evaluation of ConfTS
#
# IMPORTANT:
#
# We now use the REAL independent conformal set
# of 2000 samples.
#
# This is the same procedure used for Baseline and TS.
# ============================================================

confTS_tau, confTS_coverage, confTS_aps_size = evaluate_aps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    best_confTS_T,
    alpha,
    u_conformal,
    u_test
)


# ============================================================
# 19. Baseline vs TS vs ConfTS
# ============================================================

print(
    "\n================ CIFAR-100 APS FINAL ================"
)


print(
    f"{'Method':<12}"
    f"{'T':>8}"
    f"{'ECE':>10}"
    f"{'Tau':>10}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)


print(
    f"{'Baseline':<12}"
    f"{baseline_T:>8.3f}"
    f"{baseline_ece:>10.4f}"
    f"{baseline_tau:>10.4f}"
    f"{baseline_coverage:>12.4f}"
    f"{baseline_aps_size:>12.4f}"
)


print(
    f"{'TS':<12}"
    f"{best_T:>8.3f}"
    f"{ts_ece:>10.4f}"
    f"{ts_tau:>10.4f}"
    f"{ts_coverage:>12.4f}"
    f"{ts_aps_size:>12.4f}"
)


print(
    f"{'ConfTS':<12}"
    f"{best_confTS_T:>8.3f}"
    f"{confTS_ece:>10.4f}"
    f"{confTS_tau:>10.4f}"
    f"{confTS_coverage:>12.4f}"
    f"{confTS_aps_size:>12.4f}"
)


# ============================================================
# 20. Diagnose small-temperature numerical precision
# ============================================================

def diagnose_temperature(logits, labels, temperature):

    probabilities = torch.softmax(
        logits / temperature,
        dim=1
    )

    zero_count = (
        probabilities == 0
    ).sum().item()

    total_probabilities = probabilities.numel()

    zero_percent = (
        zero_count
        / total_probabilities
        * 100
    )


    max_probabilities = probabilities.max(
        dim=1
    ).values

    mean_max_probability = (
        max_probabilities.mean().item()
    )


    scores = nonrandomized_aps_scores(
        logits,
        labels,
        temperature
    )

    almost_one = (
        scores >= 0.999999
    ).float().mean().item()


    print(
        f"T={temperature:<5.2f} | "
        f"zeros={zero_count:<8d} "
        f"({zero_percent:>6.3f}%) | "
        f"mean max prob={mean_max_probability:.4f} | "
        f"true scores ~1={almost_one:.4f}"
    )


print(
    "\n================ TEMPERATURE DIAGNOSTIC ================"
)

for T in [
    0.30,
    0.40,
    0.50,
    0.60,
    0.80,
    1.00,
    1.40
]:

    diagnose_temperature(
        tuning_logits,
        tuning_labels,
        T
    )


# ============================================================
# 21. Numerically safer ConfTS search
#
# We exclude T < 0.40 because our diagnostic showed
# exact zero probabilities beginning at T = 0.30.
# ============================================================

safe_temperatures = torch.arange(
    0.40,
    1.51,
    0.025
)

safe_best_T = None
safe_best_loss = float("inf")
safe_best_tuning_tau = None


for T_tensor in safe_temperatures:

    T = T_tensor.item()

    # ------------------------------------
    # Compute tau(T)
    # ------------------------------------

    tau_scores = nonrandomized_aps_scores(
        confTS_tau_logits,
        confTS_tau_labels,
        T
    )

    tau_T = conformal_threshold(
        tau_scores,
        alpha
    )


    # ------------------------------------
    # Compute ConfTS loss
    # ------------------------------------

    loss_scores = nonrandomized_aps_scores(
        confTS_loss_logits,
        confTS_loss_labels,
        T
    )

    loss = torch.mean(
        (
            tau_T
            - loss_scores
        ) ** 2
    ).item()


    if loss < safe_best_loss:

        safe_best_loss = loss
        safe_best_T = T
        safe_best_tuning_tau = tau_T


print(
    "\n================ SAFE ConfTS ================"
)

print(
    f"Best safe ConfTS T:    {safe_best_T:.3f}"
)

print(
    f"Best safe ConfTS loss: {safe_best_loss:.6f}"
)

print(
    f"Tuning tau:            {safe_best_tuning_tau:.4f}"
)


# ============================================================
# 22. Evaluate safe ConfTS on final test set
# ============================================================

safe_confTS_probs = probabilities_from_logits(
    test_logits,
    safe_best_T
)

safe_confTS_ece = calculate_ece(
    safe_confTS_probs,
    test_labels
)


safe_tau, safe_coverage, safe_aps_size = evaluate_aps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    safe_best_T,
    alpha,
    u_conformal,
    u_test
)


print(
    "\n================ SAFE FINAL COMPARISON ================"
)

print(
    f"{'Method':<12}"
    f"{'T':>8}"
    f"{'ECE':>10}"
    f"{'Tau':>10}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)


print(
    f"{'Baseline':<12}"
    f"{baseline_T:>8.3f}"
    f"{baseline_ece:>10.4f}"
    f"{baseline_tau:>10.4f}"
    f"{baseline_coverage:>12.4f}"
    f"{baseline_aps_size:>12.4f}"
)


print(
    f"{'TS':<12}"
    f"{best_T:>8.3f}"
    f"{ts_ece:>10.4f}"
    f"{ts_tau:>10.4f}"
    f"{ts_coverage:>12.4f}"
    f"{ts_aps_size:>12.4f}"
)


print(
    f"{'ConfTS-safe':<12}"
    f"{safe_best_T:>8.3f}"
    f"{safe_confTS_ece:>10.4f}"
    f"{safe_tau:>10.4f}"
    f"{safe_coverage:>12.4f}"
    f"{safe_aps_size:>12.4f}"
)


# ============================================================
# 23. RAPS
# ============================================================

k_reg = 1
raps_lambda = 0.001


def raps_calibration_scores(
    logits,
    labels,
    temperature,
    u,
    k_reg,
    raps_lambda
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

    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )


    true_positions = (
        sorted_indices
        == labels.unsqueeze(1)
    ).nonzero(as_tuple=False)[:, 1]


    rows = torch.arange(
        len(labels)
    )


    true_probs = sorted_probs[
        rows,
        true_positions
    ]

    before_true = cumulative_before[
        rows,
        true_positions
    ]


    # true_positions is 0-based.
    #
    # RAPS rank is conceptually 1-based:
    #
    # position 0 → rank 1
    # position 1 → rank 2
    # etc.
    true_ranks = (
        true_positions + 1
    ).float()


    penalty = (
        raps_lambda
        * torch.clamp(
            true_ranks - k_reg,
            min=0
        )
    )


    scores = (
        before_true
        + u * true_probs
        + penalty
    )


    return scores


def evaluate_raps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    temperature,
    alpha,
    u_conformal,
    u_test,
    k_reg,
    raps_lambda
):

    # --------------------------------------------------------
    # Calibration scores
    # --------------------------------------------------------

    calibration_scores = raps_calibration_scores(
        conformal_logits,
        conformal_labels,
        temperature,
        u_conformal,
        k_reg,
        raps_lambda
    )


    tau = conformal_threshold(
        calibration_scores,
        alpha
    )


    # --------------------------------------------------------
    # Test probabilities
    # --------------------------------------------------------

    probabilities = torch.softmax(
        test_logits / temperature,
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

    cumulative_before = (
        cumulative_probs
        - sorted_probs
    )


    # --------------------------------------------------------
    # RAPS penalty for every candidate rank
    # --------------------------------------------------------

    number_of_classes = probabilities.shape[1]

    ranks = torch.arange(
        1,
        number_of_classes + 1
    ).float()


    penalties = (
        raps_lambda
        * torch.clamp(
            ranks - k_reg,
            min=0
        )
    )


    # --------------------------------------------------------
    # Candidate RAPS scores
    # --------------------------------------------------------

    sorted_scores = (
        cumulative_before
        + u_test * sorted_probs
        + penalties.unsqueeze(0)
    )


    included_sorted = (
        sorted_scores <= tau
    )


    # --------------------------------------------------------
    # Average set size
    # --------------------------------------------------------

    set_sizes = included_sorted.sum(
        dim=1
    )

    average_size = (
        set_sizes.float()
        .mean()
        .item()
    )


    # --------------------------------------------------------
    # Coverage
    # --------------------------------------------------------

    included_original = torch.zeros_like(
        included_sorted
    )

    included_original.scatter_(
        1,
        sorted_indices,
        included_sorted
    )


    rows = torch.arange(
        len(test_labels)
    )


    covered = included_original[
        rows,
        test_labels
    ]


    coverage = (
        covered.float()
        .mean()
        .item()
    )


    return tau, coverage, average_size


# ============================================================
# 24. Evaluate all three methods with RAPS
# ============================================================

baseline_raps_tau, baseline_raps_cov, baseline_raps_size = evaluate_raps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    baseline_T,
    alpha,
    u_conformal,
    u_test,
    k_reg,
    raps_lambda
)


ts_raps_tau, ts_raps_cov, ts_raps_size = evaluate_raps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    best_T,
    alpha,
    u_conformal,
    u_test,
    k_reg,
    raps_lambda
)


confTS_raps_tau, confTS_raps_cov, confTS_raps_size = evaluate_raps(
    conformal_logits,
    conformal_labels,
    test_logits,
    test_labels,
    safe_best_T,
    alpha,
    u_conformal,
    u_test,
    k_reg,
    raps_lambda
)


print(
    "\n================ CIFAR-100 RAPS ================"
)

print(
    f"k_reg = {k_reg}, lambda = {raps_lambda}"
)

print(
    f"{'Method':<12}"
    f"{'T':>8}"
    f"{'ECE':>10}"
    f"{'Tau':>10}"
    f"{'Coverage':>12}"
    f"{'RAPS Size':>12}"
)


print(
    f"{'Baseline':<12}"
    f"{baseline_T:>8.3f}"
    f"{baseline_ece:>10.4f}"
    f"{baseline_raps_tau:>10.4f}"
    f"{baseline_raps_cov:>12.4f}"
    f"{baseline_raps_size:>12.4f}"
)


print(
    f"{'TS':<12}"
    f"{best_T:>8.3f}"
    f"{ts_ece:>10.4f}"
    f"{ts_raps_tau:>10.4f}"
    f"{ts_raps_cov:>12.4f}"
    f"{ts_raps_size:>12.4f}"
)


print(
    f"{'ConfTS':<12}"
    f"{safe_best_T:>8.3f}"
    f"{safe_confTS_ece:>10.4f}"
    f"{confTS_raps_tau:>10.4f}"
    f"{confTS_raps_cov:>12.4f}"
    f"{confTS_raps_size:>12.4f}"
)

# ============================================================
# STEP 24A — CIFAR-100 TEMPERATURE SWEEP
# ============================================================

import matplotlib.pyplot as plt

sweep_temperatures = np.arange(
    0.4,
    1.61,
    0.1
)

sweep_sizes = []
sweep_coverages = []

print(
    "\n================ STEP 24A: APS TEMPERATURE SWEEP ================"
)

print(
    f"{'T':>6}"
    f"{'Tau':>12}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)

print("-" * 42)


for T in sweep_temperatures:

    # Use exactly the SAME APS implementation
    # that we used above for Baseline, TS and ConfTS.
    tau, coverage, average_size = evaluate_aps(
        conformal_logits,
        conformal_labels,
        test_logits,
        test_labels,
        float(T),
        alpha,
        u_conformal,
        u_test
    )

    sweep_coverages.append(
        coverage
    )

    sweep_sizes.append(
        average_size
    )

    print(
        f"{T:>6.2f}"
        f"{tau:>12.4f}"
        f"{coverage:>12.4f}"
        f"{average_size:>12.4f}"
    )


# ============================================================
# PLOT
# ============================================================

plt.figure()

plt.plot(
    sweep_temperatures,
    sweep_sizes,
    marker="o"
)

plt.xlabel("Temperature")

plt.ylabel(
    "Average APS Prediction-Set Size"
)

plt.title(
    "CIFAR-100: Temperature vs APS Set Size"
)

plt.grid(True)

plt.show()
