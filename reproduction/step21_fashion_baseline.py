import math
import torch


torch.manual_seed(42)

alpha = 0.1
temperature = 1.0


# ==========================================
# 1. LOAD SAVED LOGITS
# ==========================================

data = torch.load("fashion_mnist_logits.pt")

tuning_logits = data["tuning_logits"]
tuning_labels = data["tuning_labels"]

conformal_logits = data["conformal_logits"]
conformal_labels = data["conformal_labels"]

test_logits = data["test_logits"]
test_labels = data["test_labels"]


print("================ LOADED ================")

print("Tuning:", tuning_logits.shape)
print("Conformal:", conformal_logits.shape)
print("Test:", test_logits.shape)


# ==========================================
# 2. SOFTMAX WITH TEMPERATURE
# ==========================================

conformal_probs = torch.softmax(
    conformal_logits / temperature,
    dim=1
)

test_probs = torch.softmax(
    test_logits / temperature,
    dim=1
)


# ==========================================
# 3. ECE
# ==========================================

def calculate_ece(probabilities, labels, n_bins=15):

    confidences, predictions = probabilities.max(dim=1)

    correct = predictions.eq(labels)

    ece = 0.0

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

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

            avg_confidence = confidences[in_bin].mean().item()

            accuracy = correct[in_bin].float().mean().item()

            fraction = count / len(labels)

            ece += fraction * abs(
                accuracy - avg_confidence
            )

    return ece


baseline_ece = calculate_ece(
    test_probs,
    test_labels
)


# ==========================================
# 4. RANDOM NUMBERS FOR APS
# ==========================================

# Fix these random numbers so later
# Baseline / TS / ConfTS use the SAME randomness.

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
# 5. RANDOMIZED APS CALIBRATION SCORES
# ==========================================

sorted_probs, sorted_indices = torch.sort(
    conformal_probs,
    descending=True,
    dim=1
)

cumulative_probs = torch.cumsum(
    sorted_probs,
    dim=1
)


# Find the rank of the true class
matches = (
    sorted_indices
    == conformal_labels.unsqueeze(1)
)

true_ranks = matches.float().argmax(dim=1)


row_indices = torch.arange(
    len(conformal_labels)
)


true_probs = sorted_probs[
    row_indices,
    true_ranks
]


cumulative_at_true = cumulative_probs[
    row_indices,
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


# ==========================================
# 6. CONFORMAL THRESHOLD
# ==========================================

n = len(aps_scores)

k = math.ceil(
    (n + 1) * (1 - alpha)
)

tau = torch.kthvalue(
    aps_scores,
    k
).values.item()


print("\n================ APS CALIBRATION ================")

print("Alpha:", alpha)
print("Target coverage:", 1 - alpha)
print("Number of calibration samples:", n)
print("Quantile position k:", k)
print(f"APS threshold tau: {tau:.6f}")


# ==========================================
# 7. BUILD TEST PREDICTION SETS
# ==========================================

sorted_test_probs, sorted_test_indices = torch.sort(
    test_probs,
    descending=True,
    dim=1
)

test_cumulative = torch.cumsum(
    sorted_test_probs,
    dim=1
)


# Probability accumulated BEFORE each class
previous_test_sum = (
    test_cumulative
    - sorted_test_probs
)


# Same random u for all candidate labels
# belonging to one test sample.
sorted_candidate_scores = (
    previous_test_sum
    + test_u.unsqueeze(1) * sorted_test_probs
)


# Convert scores back to original class order
candidate_scores = torch.empty_like(
    sorted_candidate_scores
)

candidate_scores.scatter_(
    1,
    sorted_test_indices,
    sorted_candidate_scores
)


prediction_sets = (
    candidate_scores <= tau
)


# ==========================================
# 8. COVERAGE
# ==========================================

row_indices = torch.arange(
    len(test_labels)
)

true_label_in_set = prediction_sets[
    row_indices,
    test_labels
]

coverage = (
    true_label_in_set
    .float()
    .mean()
    .item()
)


# ==========================================
# 9. AVERAGE SET SIZE
# ==========================================

set_sizes = prediction_sets.sum(dim=1)

average_set_size = (
    set_sizes
    .float()
    .mean()
    .item()
)


# ==========================================
# 10. FINAL BASELINE RESULT
# ==========================================

print("\n================ BASELINE T=1 ================")

print(f"ECE:                  {baseline_ece:.4f}")
print(f"APS coverage:         {coverage:.4f}")
print(f"APS average set size: {average_set_size:.4f}")


# ==========================================
# 11. A FEW EXAMPLES
# ==========================================

print("\n================ EXAMPLE SETS ================")

for i in range(5):

    included_classes = torch.where(
        prediction_sets[i]
    )[0].tolist()

    print(
        f"Sample {i}: "
        f"true={test_labels[i].item()} | "
        f"set={included_classes} | "
        f"size={len(included_classes)}"
    )