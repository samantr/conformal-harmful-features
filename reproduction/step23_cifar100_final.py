import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split


# ============================================================
# 1. Reproducibility
# ============================================================

torch.manual_seed(42)


# ============================================================
# 2. CIFAR-100 normalization
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])


test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761)
    )
])


# ============================================================
# 3. Download CIFAR-100
# ============================================================

train_dataset = datasets.CIFAR100(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)


full_test_dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)


print("\n================ CIFAR-100 ================")

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Original test images:",
    len(full_test_dataset)
)

print(
    "Number of classes:",
    len(train_dataset.classes)
)


# ============================================================
# 4. Split the original 10,000 test images
#
# 2,000 tuning
# 2,000 conformal calibration
# 6,000 final test
# ============================================================

all_indices = list(
    range(len(full_test_dataset))
)

all_labels = full_test_dataset.targets


# First:
#
# 4000 calibration-related
# 6000 final test
#
calibration_indices, test_indices = train_test_split(
    all_indices,
    test_size=6000,
    random_state=42,
    stratify=all_labels
)


calibration_labels = [
    all_labels[i]
    for i in calibration_indices
]


# Second:
#
# 2000 tuning
# 2000 conformal
#
tuning_indices, conformal_indices = train_test_split(
    calibration_indices,
    test_size=2000,
    random_state=42,
    stratify=calibration_labels
)


tuning_dataset = Subset(
    full_test_dataset,
    tuning_indices
)

conformal_dataset = Subset(
    full_test_dataset,
    conformal_indices
)

test_dataset = Subset(
    full_test_dataset,
    test_indices
)


print("\n================ SPLITS ================")

print(
    "Training:",
    len(train_dataset)
)

print(
    "Tuning:",
    len(tuning_dataset)
)

print(
    "Conformal calibration:",
    len(conformal_dataset)
)

print(
    "Final test:",
    len(test_dataset)
)


# ============================================================
# 5. DataLoaders
# ============================================================

batch_size = 128


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

tuning_loader = DataLoader(
    tuning_dataset,
    batch_size=batch_size,
    shuffle=False
)

conformal_loader = DataLoader(
    conformal_dataset,
    batch_size=batch_size,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


# ============================================================
# 6. Manageable model: ResNet18
#
# CIFAR images are only 32x32.
#
# Standard ResNet18 was designed for much larger ImageNet
# images, so we make its first layer CIFAR-friendly.
# ============================================================

model = models.resnet18(
    weights=None
)


# Original:
# 7x7 convolution with stride 2
#
# CIFAR version:
# 3x3 convolution with stride 1
#
model.conv1 = nn.Conv2d(
    3,
    64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False
)


# Do not immediately shrink our tiny 32x32 image.
model.maxpool = nn.Identity()


# CIFAR-100 has 100 classes.
model.fc = nn.Linear(
    model.fc.in_features,
    100
)


# ============================================================
# 7. Tiny sanity check
# ============================================================

images, labels = next(
    iter(train_loader)
)

print("\n================ ONE BATCH ================")

print(
    "Images:",
    images.shape
)

print(
    "Labels:",
    labels.shape
)


with torch.no_grad():

    logits = model(images)


print(
    "Logits:",
    logits.shape
)

print(
    "\nOne image produces",
    logits.shape[1],
    "logits."
)


# ============================================================
# 8. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n================ DEVICE ================")
print("Using:", device)

model = model.to(device)


# ============================================================
# 9. Training setup
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005
)


# Reduce the learning rate during training.
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[30, 40],
    gamma=0.2
)


num_epochs = 50

model_path = "cifar100_resnet18.pth"


# ============================================================
# 10. Train or load saved model
# ============================================================

if os.path.exists(model_path):

    print("\nSaved model found.")
    print("Loading:", model_path)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

else:

    print("\n================ TRAINING ================")

    for epoch in range(num_epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # ----------------------------
            # Forward
            # ----------------------------

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )


            # ----------------------------
            # Backward
            # ----------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # ----------------------------
            # Statistics
            # ----------------------------

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


        scheduler.step()


        epoch_loss = total_loss / total

        epoch_accuracy = correct / total


        print(
            f"Epoch {epoch + 1:2d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Accuracy: {epoch_accuracy:.4f}"
        )


    # Save it so we do NOT train again next time.
    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "\nModel saved to:",
        model_path
    )


# ============================================================
# 11. Extract and cache logits
#
# We do this ONCE.
#
# After this, TS / ConfTS / APS / RAPS can work directly
# with saved logits without running ResNet18 again.
# ============================================================

logits_cache_path = "cifar100_logits_cache.pt"


def collect_logits(loader):

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            logits = model(images)

            all_logits.append(
                logits.cpu()
            )

            all_labels.append(
                labels.cpu()
            )

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    return all_logits, all_labels


# ============================================================
# Load cache if it already exists
# ============================================================

if os.path.exists(logits_cache_path):

    print("\nSaved logits found.")
    print("Loading:", logits_cache_path)

    cache = torch.load(
        logits_cache_path,
        map_location="cpu"
    )

    tuning_logits = cache["tuning_logits"]
    tuning_labels = cache["tuning_labels"]

    conformal_logits = cache["conformal_logits"]
    conformal_labels = cache["conformal_labels"]

    test_logits = cache["test_logits"]
    test_labels = cache["test_labels"]


# ============================================================
# Otherwise calculate them once
# ============================================================

else:

    print("\n================ EXTRACTING LOGITS ================")

    print("Tuning set...")

    tuning_logits, tuning_labels = collect_logits(
        tuning_loader
    )


    print("Conformal calibration set...")

    conformal_logits, conformal_labels = collect_logits(
        conformal_loader
    )


    print("Final test set...")

    test_logits, test_labels = collect_logits(
        test_loader
    )


    torch.save(
        {
            "tuning_logits": tuning_logits,
            "tuning_labels": tuning_labels,

            "conformal_logits": conformal_logits,
            "conformal_labels": conformal_labels,

            "test_logits": test_logits,
            "test_labels": test_labels
        },

        logits_cache_path
    )


    print(
        "\nSaved logits to:",
        logits_cache_path
    )


# ============================================================
# 12. Inspect cached data
# ============================================================

print("\n================ CACHED LOGITS ================")

print(
    "Tuning logits:",
    tuning_logits.shape
)

print(
    "Conformal logits:",
    conformal_logits.shape
)

print(
    "Test logits:",
    test_logits.shape
)


# ============================================================
# 13. Accuracy using ONLY saved logits
# ============================================================

test_predictions = test_logits.argmax(
    dim=1
)

test_accuracy = (
    test_predictions == test_labels
).float().mean().item()


print("\n================ CLASSIFIER ================")

print(
    f"Final test accuracy: {test_accuracy:.4f}"
)