import math
import torch


alpha = 0.1

T_baseline = 1.00
T_ts = 1.10
T_confts = 0.425


# ==========================================
# 1. LOAD SAVED LOGITS
# ==========================================

data = torch.load("fashion_mnist_logits.pt")

conformal_logits = data["conformal_logits"]
conformal_labels = data["conformal_labels"]

test_logits = data["test_logits"]
test_labels = data["test_labels"]


# ==========================================
# 2. FIX RANDOMIZED APS VALUES
# ==========================================

generator = torch.Generator().manual_seed(12345)

conformal_u = torch.rand(
    len(conformal_labels),
    generator=generator
)

test_u = torch.rand(
    len(test_labels),
    generator=generator
)


# ==========================================
# 3. ECE
# ==========================================

def calculate_ece(probabilities, labels, n_bins=15):

    confidences, predictions = probabilities.max(dim=1)

    correct = predictions.eq(labels)

    ece = 0.0

    boundaries = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    for i in range(n_bins):

        lower = boundaries[i]
        upper = boundaries[i + 1]

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

        count = in_bin.sum().item()

        if count > 0:

            avg_confidence = (
                confidences[in_bin]
                .mean()
                .item()
            )

            accuracy = (
                correct[in_bin]
                .float()
                .mean()
                .item()
            )

            fraction = count / len(labels)

            ece += (
                fraction
                * abs(accuracy - avg_confidence)
            )

    return ece


# ==========================================
# 4. EVALUATE ONE TEMPERATURE WITH APS
# ==========================================

def evaluate_temperature(T):

    # --------------------------------------
    # Softmax
    # --------------------------------------

    conformal_probs = torch.softmax(
        conformal_logits / T,
        dim=1
    )

    test_probs = torch.softmax(
        test_logits / T,
        dim=1
    )


    # --------------------------------------
    # ECE
    # --------------------------------------

    ece = calculate_ece(
        test_probs,
        test_labels
    )


    # ======================================
    # APS CALIBRATION
    # ======================================

    sorted_probs, sorted_indices = torch.sort(
        conformal_probs,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )


    # Find true-label rank
    matches = (
        sorted_indices
        == conformal_labels.unsqueeze(1)
    )

    true_ranks = (
        matches.float()
        .argmax(dim=1)
    )

    rows = torch.arange(
        len(conformal_labels)
    )


    # Probability of true class
    true_probs = sorted_probs[
        rows,
        true_ranks
    ]


    # Cumulative probability including true class
    cumulative_at_true = cumulative_probs[
        rows,
        true_ranks
    ]


    # Probability accumulated before true class
    previous_sum = (
        cumulative_at_true
        - true_probs
    )


    # Randomized APS score
    calibration_scores = (
        previous_sum
        + conformal_u * true_probs
    )


    # ======================================
    # CONFORMAL THRESHOLD
    # ======================================

    n = len(calibration_scores)

    k = math.ceil(
        (n + 1) * (1 - alpha)
    )

    tau = torch.kthvalue(
        calibration_scores,
        k
    ).values.item()


    # ======================================
    # TEST APS SCORES
    # ======================================

    sorted_test_probs, sorted_test_indices = torch.sort(
        test_probs,
        descending=True,
        dim=1
    )

    cumulative_test = torch.cumsum(
        sorted_test_probs,
        dim=1
    )

    previous_test_sum = (
        cumulative_test
        - sorted_test_probs
    )


    # Randomized candidate-label scores
    sorted_candidate_scores = (
        previous_test_sum
        + test_u.unsqueeze(1)
        * sorted_test_probs
    )


    # Put scores back in original class order
    candidate_scores = torch.empty_like(
        sorted_candidate_scores
    )

    candidate_scores.scatter_(
        1,
        sorted_test_indices,
        sorted_candidate_scores
    )


    # ======================================
    # PREDICTION SETS
    # ======================================

    prediction_sets = (
        candidate_scores <= tau
    )


    # ======================================
    # COVERAGE
    # ======================================

    rows = torch.arange(
        len(test_labels)
    )

    coverage = (
        prediction_sets[
            rows,
            test_labels
        ]
        .float()
        .mean()
        .item()
    )


    # ======================================
    # AVERAGE SET SIZE
    # ======================================

    set_sizes = prediction_sets.sum(
        dim=1
    )

    average_size = (
        set_sizes
        .float()
        .mean()
        .item()
    )


    return {
        "T": T,
        "ECE": ece,
        "tau": tau,
        "coverage": coverage,
        "size": average_size
    }


# ==========================================
# 5. RUN ALL THREE METHODS
# ==========================================

baseline = evaluate_temperature(
    T_baseline
)

ts = evaluate_temperature(
    T_ts
)

confts = evaluate_temperature(
    T_confts
)


# ==========================================
# 6. FINAL TABLE
# ==========================================

print(
    "================ FASHION-MNIST APS RESULTS ================"
)

print(
    f"{'Method':<12}"
    f"{'T':>10}"
    f"{'ECE':>12}"
    f"{'Tau':>12}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)


for name, result in [
    ("Baseline", baseline),
    ("TS", ts),
    ("ConfTS", confts)
]:

    print(
        f"{name:<12}"
        f"{result['T']:>10.3f}"
        f"{result['ECE']:>12.4f}"
        f"{result['tau']:>12.4f}"
        f"{result['coverage']:>12.4f}"
        f"{result['size']:>12.4f}"
    )