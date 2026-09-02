import torch
import torch.nn.functional as F


# ==========================================
# 1. LOAD LOGITS
# ==========================================

data = torch.load("fashion_mnist_logits.pt")

tuning_logits = data["tuning_logits"]
tuning_labels = data["tuning_labels"]

conformal_logits = data["conformal_logits"]
conformal_labels = data["conformal_labels"]

test_logits = data["test_logits"]
test_labels = data["test_labels"]


# ==========================================
# 2. TEMPERATURE GRID
# ==========================================

temperatures = torch.arange(
    0.4,
    2.01,
    0.05
)


best_temperature = None
best_nll = float("inf")


print("================ TEMPERATURE SCALING ================")

for T in temperatures:

    T = T.item()

    scaled_logits = tuning_logits / T

    nll = F.cross_entropy(
        scaled_logits,
        tuning_labels
    ).item()

    print(
        f"T = {T:.2f} | "
        f"Tuning NLL = {nll:.6f}"
    )

    if nll < best_nll:

        best_nll = nll
        best_temperature = T


print("\n================ BEST TS ================")

print(f"Best temperature: {best_temperature:.2f}")
print(f"Best tuning NLL:  {best_nll:.6f}")


# Compare with baseline T=1

baseline_nll = F.cross_entropy(
    tuning_logits,
    tuning_labels
).item()

print(f"Baseline T=1 NLL: {baseline_nll:.6f}")