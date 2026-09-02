import math
import torch


torch.manual_seed(42)

alpha = 0.1

T_baseline = 1.0
T_ts = 1.10


# ==========================================
# 1. LOAD SAVED LOGITS
# ==========================================

data = torch.load("fashion_mnist_logits.pt")

conformal_logits = data["conformal_logits"]
conformal_labels = data["conformal_labels"]

test_logits = data["test_logits"]
test_labels = data["test_labels"]


# ==========================================
# 2. FIX APS RANDOMNESS
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
# 4. EVALUATE ONE TEMPERATURE
# ==========================================

def evaluate_temperature(T):

    # --------------------------------------
    # probabilities
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


    # --------------------------------------
    # APS CALIBRATION SCORES
    # --------------------------------------

    sorted_probs, sorted_indices = torch.sort(
        conformal_probs,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    matches = (
        sorted_indices
        == conformal_labels.unsqueeze(1)
    )

    true_ranks = matches.float().argmax(
        dim=1
    )

    rows = torch.arange(
        len(conformal_labels)
    )

    true_probs = sorted_probs[
        rows,
        true_ranks
    ]

    cumulative_at_true = cumulative_probs[
        rows,
        true_ranks
    ]

    previous_sum = (
        cumulative_at_true
        - true_probs
    )

    aps_scores = (
        previous_sum
        + conformal_u * true_probs
    )


    # --------------------------------------
    # TAU
    # --------------------------------------

    n = len(aps_scores)

    k = math.ceil(
        (n + 1) * (1 - alpha)
    )

    tau = torch.kthvalue(
        aps_scores,
        k
    ).values.item()


    # --------------------------------------
    # TEST APS SCORES
    # --------------------------------------

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

    sorted_candidate_scores = (
        previous_test_sum
        + test_u.unsqueeze(1)
        * sorted_test_probs
    )


    # Return scores to original class order

    candidate_scores = torch.empty_like(
        sorted_candidate_scores
    )

    candidate_scores.scatter_(
        1,
        sorted_test_indices,
        sorted_candidate_scores
    )


    # --------------------------------------
    # PREDICTION SETS
    # --------------------------------------

    prediction_sets = (
        candidate_scores <= tau
    )


    # --------------------------------------
    # COVERAGE
    # --------------------------------------

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


    # --------------------------------------
    # AVERAGE SIZE
    # --------------------------------------

    average_size = (
        prediction_sets
        .sum(dim=1)
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
# 5. RUN BOTH
# ==========================================

baseline = evaluate_temperature(
    T_baseline
)

ts = evaluate_temperature(
    T_ts
)


# ==========================================
# 6. RESULTS
# ==========================================

print("================ RESULTS ================")

print(
    f"{'Method':<12}"
    f"{'T':>8}"
    f"{'ECE':>12}"
    f"{'Tau':>12}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)

print(
    f"{'Baseline':<12}"
    f"{baseline['T']:>8.2f}"
    f"{baseline['ECE']:>12.4f}"
    f"{baseline['tau']:>12.4f}"
    f"{baseline['coverage']:>12.4f}"
    f"{baseline['size']:>12.4f}"
)

print(
    f"{'TS':<12}"
    f"{ts['T']:>8.2f}"
    f"{ts['ECE']:>12.4f}"
    f"{ts['tau']:>12.4f}"
    f"{ts['coverage']:>12.4f}"
    f"{ts['size']:>12.4f}"
)