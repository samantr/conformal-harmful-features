import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import math
import numpy as np
import torch.nn.functional as F

# ==========================================
# 1. DEVICE
# ==========================================
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ==========================================
# 2. CIFAR-10
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),

    # Rough CIFAR-10 normalization
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


train_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_full = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


print("Original training samples:", len(train_full))
print("Original test samples:", len(test_full))
print("Classes:", train_full.classes)


# ==========================================
# 3. SPLITS
# ==========================================

generator = torch.Generator().manual_seed(42)

# CIFAR train set:
# 40,000 model training
# 10,000 tuning
train_set, tuning_set = random_split(
    train_full,
    [40000, 10000],
    generator=generator
)

# CIFAR official test set:
# 5,000 conformal calibration
# 5,000 final test
conformal_set, test_set = random_split(
    test_full,
    [5000, 5000],
    generator=generator
)


print("\n================ SPLITS ================")
print("Training:", len(train_set))
print("Tuning:", len(tuning_set))
print("Conformal calibration:", len(conformal_set))
print("Test:", len(test_set))


# ==========================================
# 4. DATA LOADERS
# ==========================================

batch_size = 128

train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True
)

tuning_loader = DataLoader(
    tuning_set,
    batch_size=batch_size,
    shuffle=False
)

conformal_loader = DataLoader(
    conformal_set,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False
)


# ==========================================
# 5. SMALL CNN
# ==========================================

class SmallCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # 32 x 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 8 x 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SmallCNN().to(device)

print("\nModel:")
print(model)


# ==========================================
# 6. TRAINING
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 12


print("\n================ TRAINING ================")

for epoch in range(1, epochs + 1):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        predictions = logits.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    print(
        f"Epoch {epoch:2d} | "
        f"Loss: {average_loss:.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================
# 7. FINAL TEST ACCURACY
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        predictions = logits.argmax(dim=1)

        correct += (predictions == y).sum().item()
        total += y.size(0)


test_accuracy = correct / total

print("\n================ RESULT ================")
print(f"Test accuracy: {test_accuracy:.4f}")

# ==========================================
# 8. COLLECT LOGITS
# ==========================================

@torch.no_grad()
def collect_logits(loader):

    model.eval()

    all_logits = []
    all_labels = []

    for x, y in loader:

        x = x.to(device)

        logits = model(x)

        all_logits.append(logits.cpu())
        all_labels.append(y)

    return torch.cat(all_logits), torch.cat(all_labels)


print("\n================ COLLECTING LOGITS ================")

tuning_logits, tuning_labels = collect_logits(tuning_loader)
conformal_logits, conformal_labels = collect_logits(conformal_loader)
test_logits, test_labels = collect_logits(test_loader)

print("Tuning logits:", tuning_logits.shape)
print("Conformal logits:", conformal_logits.shape)
print("Test logits:", test_logits.shape)

# ==========================================
# 9. APS FUNCTIONS
# ==========================================

alpha = 0.10


def conformal_quantile(scores, alpha):

    n = len(scores)

    k = math.ceil((n + 1) * (1 - alpha))
    k = min(k, n)

    sorted_scores = torch.sort(scores).values

    return sorted_scores[k - 1].item()


def nonrandomized_aps_true_scores(probabilities, labels):

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=1)

    # Find the rank of the true class
    true_positions = (
        sorted_indices == labels.unsqueeze(1)
    ).long().argmax(dim=1)

    true_scores = cumulative_probs.gather(
        1,
        true_positions.unsqueeze(1)
    ).squeeze(1)

    return true_scores


def randomized_aps_true_scores(probabilities, labels, seed):

    generator = torch.Generator().manual_seed(seed)

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(sorted_probs, dim=1)

    cumulative_before = cumulative_probs - sorted_probs

    u = torch.rand(
        probabilities.shape[0],
        generator=generator
    )

    randomized_scores = (
        cumulative_before
        + u.unsqueeze(1) * sorted_probs
    )

    true_positions = (
        sorted_indices == labels.unsqueeze(1)
    ).long().argmax(dim=1)

    true_scores = randomized_scores.gather(
        1,
        true_positions.unsqueeze(1)
    ).squeeze(1)

    return true_scores


def evaluate_randomized_aps(
    probabilities,
    labels,
    tau,
    seed
):

    generator = torch.Generator().manual_seed(seed)

    sorted_probs, sorted_indices = torch.sort(
        probabilities,
        descending=True,
        dim=1
    )

    cumulative_probs = torch.cumsum(
        sorted_probs,
        dim=1
    )

    cumulative_before = cumulative_probs - sorted_probs

    u = torch.rand(
        probabilities.shape[0],
        generator=generator
    )

    candidate_scores = (
        cumulative_before
        + u.unsqueeze(1) * sorted_probs
    )

    # Include classes whose APS score <= tau
    included = candidate_scores <= tau

    set_sizes = included.sum(dim=1)

    true_positions = (
        sorted_indices == labels.unsqueeze(1)
    ).long().argmax(dim=1)

    true_scores = candidate_scores.gather(
        1,
        true_positions.unsqueeze(1)
    ).squeeze(1)

    coverage = (true_scores <= tau).float().mean().item()
    average_size = set_sizes.float().mean().item()

    return coverage, average_size

# ==========================================
# 10. ORDINARY TEMPERATURE SCALING
# ==========================================

temperature_grid = np.arange(
    0.20,
    3.01,
    0.05
)


best_ts_temperature = None
best_nll = float("inf")


for T in temperature_grid:

    nll = F.cross_entropy(
        tuning_logits / T,
        tuning_labels
    ).item()

    if nll < best_nll:

        best_nll = nll
        best_ts_temperature = T


print("\n================ ORDINARY TS ================")
print(f"Best T: {best_ts_temperature:.3f}")
print(f"Best tuning NLL: {best_nll:.4f}")

# ==========================================
# 11. CONFTS GRID SEARCH
# ==========================================

loss_logits = tuning_logits[:5000]
loss_labels = tuning_labels[:5000]

confTS_cal_logits = tuning_logits[5000:]
confTS_cal_labels = tuning_labels[5000:]


best_confts_temperature = None
best_confts_loss = float("inf")


for T in temperature_grid:

    loss_probs = torch.softmax(
        loss_logits / T,
        dim=1
    )

    confTS_cal_probs = torch.softmax(
        confTS_cal_logits / T,
        dim=1
    )

    calibration_scores = nonrandomized_aps_true_scores(
        confTS_cal_probs,
        confTS_cal_labels
    )

    loss_scores = nonrandomized_aps_true_scores(
        loss_probs,
        loss_labels
    )

    tau = conformal_quantile(
        calibration_scores,
        alpha
    )

    # ----------------------------------
    # Numerical safety check
    # ----------------------------------

    zero_probs = (
        (loss_probs == 0).sum().item()
        +
        (confTS_cal_probs == 0).sum().item()
    )

    # Skip temperatures where probabilities
    # have collapsed numerically or tau is
    # exactly 1.
    if zero_probs > 0 or tau >= 1.0:
        continue

    confts_loss = torch.mean(
        (tau - loss_scores) ** 2
    ).item()

    if confts_loss < best_confts_loss:

        best_confts_loss = confts_loss
        best_confts_temperature = T

print("\n================ CONFTS ================")

print(
    f"Best T: {best_confts_temperature:.3f}"
)

print(
    f"Best ConfTS loss: {best_confts_loss:.6f}"
)

print("\n================ CONFTS DIAGNOSTIC ================")

diagnostic_temperatures = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    1.00
]

print(
    f"{'T':>6}"
    f"{'Tau':>14}"
    f"{'ConfTS Loss':>16}"
    f"{'Zero Probs':>14}"
)

for T in diagnostic_temperatures:

    loss_probs = torch.softmax(
        loss_logits / T,
        dim=1
    )

    cal_probs = torch.softmax(
        confTS_cal_logits / T,
        dim=1
    )

    calibration_scores = nonrandomized_aps_true_scores(
        cal_probs,
        confTS_cal_labels
    )

    loss_scores = nonrandomized_aps_true_scores(
        loss_probs,
        loss_labels
    )

    tau = conformal_quantile(
        calibration_scores,
        alpha
    )

    confts_loss = torch.mean(
        (tau - loss_scores) ** 2
    ).item()

    zero_probs = (
        cal_probs == 0
    ).sum().item()

    print(
        f"{T:>6.2f}"
        f"{tau:>14.10f}"
        f"{confts_loss:>16.8f}"
        f"{zero_probs:>14}"
    )

# ==========================================
# 12. ECE
# ==========================================

def calculate_ece(probabilities, labels, n_bins=15):

    confidence, predictions = probabilities.max(dim=1)

    correct = predictions.eq(labels)

    ece = 0.0

    bin_edges = torch.linspace(
        0,
        1,
        n_bins + 1
    )

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        in_bin = (
            (confidence > lower)
            & (confidence <= upper)
        )

        if in_bin.sum() == 0:
            continue

        bin_accuracy = correct[in_bin].float().mean()
        bin_confidence = confidence[in_bin].mean()

        bin_weight = in_bin.float().mean()

        ece += (
            bin_weight
            * torch.abs(
                bin_accuracy - bin_confidence
            )
        )

    return ece.item()

# ==========================================
# 13. FINAL APS COMPARISON
# ==========================================

methods = {
    "Baseline": 1.0,
    "TS": best_ts_temperature,
    "ConfTS": best_confts_temperature
}


print("\n================ CIFAR-10 APS RESULTS ================")

print(
    f"{'Method':<12}"
    f"{'T':>10}"
    f"{'ECE':>12}"
    f"{'Tau':>12}"
    f"{'Coverage':>12}"
    f"{'APS Size':>12}"
)


for method_name, T in methods.items():

    # ----------------------------------
    # Probabilities
    # ----------------------------------

    conformal_probs = torch.softmax(
        conformal_logits / T,
        dim=1
    )

    test_probs = torch.softmax(
        test_logits / T,
        dim=1
    )


    # ----------------------------------
    # RANDOMIZED APS calibration scores
    # ----------------------------------

    calibration_scores = randomized_aps_true_scores(
        conformal_probs,
        conformal_labels,
        seed=123
    )


    # ----------------------------------
    # Final conformal threshold
    # ----------------------------------

    tau = conformal_quantile(
        calibration_scores,
        alpha
    )


    # ----------------------------------
    # Final test prediction sets
    # ----------------------------------

    coverage, average_size = evaluate_randomized_aps(
        test_probs,
        test_labels,
        tau,
        seed=456
    )


    # ----------------------------------
    # ECE
    # ----------------------------------

    ece = calculate_ece(
        test_probs,
        test_labels
    )


    print(
        f"{method_name:<12}"
        f"{T:>10.3f}"
        f"{ece:>12.4f}"
        f"{tau:>12.4f}"
        f"{coverage:>12.4f}"
        f"{average_size:>12.4f}"
    )

    # ==========================================
    # 14. RAPS FUNCTIONS
    # ==========================================

    k_reg = 1
    lambda_reg = 0.001


    def randomized_raps_true_scores(
            probabilities,
            labels,
            seed
    ):
        generator = torch.Generator().manual_seed(seed)

        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=1
        )

        cumulative_before = cumulative_probs - sorted_probs

        u = torch.rand(
            probabilities.shape[0],
            generator=generator
        )

        # Rank: 1, 2, 3, ..., 10
        ranks = torch.arange(
            1,
            probabilities.shape[1] + 1
        ).unsqueeze(0)

        penalty = lambda_reg * torch.clamp(
            ranks - k_reg,
            min=0
        )

        raps_scores = (
                cumulative_before
                + u.unsqueeze(1) * sorted_probs
                + penalty
        )

        true_positions = (
                sorted_indices == labels.unsqueeze(1)
        ).long().argmax(dim=1)

        true_scores = raps_scores.gather(
            1,
            true_positions.unsqueeze(1)
        ).squeeze(1)

        return true_scores


    def evaluate_randomized_raps(
            probabilities,
            labels,
            tau,
            seed
    ):
        generator = torch.Generator().manual_seed(seed)

        sorted_probs, sorted_indices = torch.sort(
            probabilities,
            descending=True,
            dim=1
        )

        cumulative_probs = torch.cumsum(
            sorted_probs,
            dim=1
        )

        cumulative_before = cumulative_probs - sorted_probs

        u = torch.rand(
            probabilities.shape[0],
            generator=generator
        )

        ranks = torch.arange(
            1,
            probabilities.shape[1] + 1
        ).unsqueeze(0)

        penalty = lambda_reg * torch.clamp(
            ranks - k_reg,
            min=0
        )

        candidate_scores = (
                cumulative_before
                + u.unsqueeze(1) * sorted_probs
                + penalty
        )

        included = candidate_scores <= tau

        set_sizes = included.sum(dim=1)

        true_positions = (
                sorted_indices == labels.unsqueeze(1)
        ).long().argmax(dim=1)

        true_scores = candidate_scores.gather(
            1,
            true_positions.unsqueeze(1)
        ).squeeze(1)

        coverage = (
                true_scores <= tau
        ).float().mean().item()

        average_size = (
            set_sizes.float().mean().item()
        )

        return coverage, average_size


    # ==========================================
    # 15. FINAL RAPS COMPARISON
    # ==========================================

    print("\n================ CIFAR-10 RAPS RESULTS ================")

    print(
        f"k_reg = {k_reg}, lambda = {lambda_reg}\n"
    )

    print(
        f"{'Method':<12}"
        f"{'T':>10}"
        f"{'ECE':>12}"
        f"{'Tau':>12}"
        f"{'Coverage':>12}"
        f"{'RAPS Size':>12}"
    )

    for method_name, T in methods.items():
        # ----------------------------------
        # Probabilities
        # ----------------------------------

        conformal_probs = torch.softmax(
            conformal_logits / T,
            dim=1
        )

        test_probs = torch.softmax(
            test_logits / T,
            dim=1
        )

        # ----------------------------------
        # RAPS calibration scores
        # ----------------------------------

        calibration_scores = randomized_raps_true_scores(
            conformal_probs,
            conformal_labels,
            seed=123
        )

        # ----------------------------------
        # RAPS threshold
        # ----------------------------------

        tau = conformal_quantile(
            calibration_scores,
            alpha
        )

        # ----------------------------------
        # Test RAPS sets
        # ----------------------------------

        coverage, average_size = evaluate_randomized_raps(
            test_probs,
            test_labels,
            tau,
            seed=456
        )

        # ----------------------------------
        # ECE
        # ----------------------------------

        ece = calculate_ece(
            test_probs,
            test_labels
        )

        print(
            f"{method_name:<12}"
            f"{T:>10.3f}"
            f"{ece:>12.4f}"
            f"{tau:>12.4f}"
            f"{coverage:>12.4f}"
            f"{average_size:>12.4f}"
        )