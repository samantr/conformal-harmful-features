import math
import torch


alpha = 0.1


# ==========================================
# 1. LOAD DATA
# ==========================================

data = torch.load("fashion_mnist_logits.pt")

tuning_logits = data["tuning_logits"]
tuning_labels = data["tuning_labels"]


# ==========================================
# 2. SPLIT TUNING SET FOR CONFTS
# ==========================================

confTS_threshold_logits = tuning_logits[:5000]
confTS_threshold_labels = tuning_labels[:5000]

confTS_loss_logits = tuning_logits[5000:]
confTS_loss_labels = tuning_labels[5000:]


print("================ CONFTS SPLIT ================")

print(
    "Threshold subset:",
    confTS_threshold_logits.shape
)

print(
    "Loss subset:",
    confTS_loss_logits.shape
)


# ==========================================
# 3. NON-RANDOMIZED TRUE-LABEL APS SCORES
# ==========================================

def nonrandomized_true_aps_scores(
    logits,
    labels,
    T
):

    probabilities = torch.softmax(
        logits / T,
        dim=1
    )

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    matches = (
        sorted_indices
        == labels.unsqueeze(1)
    )

    true_ranks = (
        matches.float()
        .argmax(dim=1)
    )

    rows = torch.arange(
        len(labels)
    )

    scores = cumulative_probs[
        rows,
        true_ranks
    ]

    return scores


# ==========================================
# 4. TRY MANY TEMPERATURES
# ==========================================

temperatures = torch.arange(
    0.15,
    0.61,
    0.025
)


best_T = None
best_loss = float("inf")
best_tau = None


print("\n================ CONFTS GRID ================")

for T in temperatures:

    T = T.item()


    # --------------------------------------
    # Scores used to calculate tau
    # --------------------------------------

    threshold_scores = (
        nonrandomized_true_aps_scores(
            confTS_threshold_logits,
            confTS_threshold_labels,
            T
        )
    )



    # --------------------------------------
    # Calculate conformal tau(T)
    # --------------------------------------

    n = len(threshold_scores)

    k = math.ceil(
        (n + 1) * (1 - alpha)
    )

    tau = torch.kthvalue(
        threshold_scores,
        k
    ).values


    # --------------------------------------
    # Scores used for ConfTS loss
    # --------------------------------------

    loss_scores = (
        nonrandomized_true_aps_scores(
            confTS_loss_logits,
            confTS_loss_labels,
            T
        )
    )


    # --------------------------------------
    # ConfTS loss:
    #
    # (tau - S(x,y))^2
    # --------------------------------------

    loss = torch.mean(
        (tau - loss_scores) ** 2
    ).item()

    # --------------------------------------
    # CHECK FOR NUMERICAL ZERO PROBABILITIES
    # --------------------------------------

    threshold_probs = torch.softmax(
        confTS_threshold_logits / T,
        dim=1
    )

    zero_count = (
            threshold_probs == 0
    ).sum().item()

    print(
        f"T = {T:.3f} | "
        f"tau = {tau.item():.6f} | "
        f"loss = {loss:.6f} | "
        f"exact zero probabilities = {zero_count}"
    )

    if loss < best_loss:
        best_loss = loss
        best_T = T
        best_tau = tau.item()

# ==========================================
# 5. RESULT
# ==========================================

print("\n================ BEST CONFTS ================")

print(f"Best T:          {best_T:.2f}")
print(f"Best ConfTS loss:{best_loss:.6f}")
print(f"Internal tau:    {best_tau:.6f}")