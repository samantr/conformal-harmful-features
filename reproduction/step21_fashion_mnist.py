import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)


# ==========================================
# 1. LOAD FASHION-MNIST
# ==========================================

transform = transforms.ToTensor()

full_train = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


print("Full training dataset:", len(full_train))
print("Test dataset:", len(test_dataset))


# ==========================================
# 2. SPLIT TRAINING DATA
# ==========================================

train_dataset, tuning_dataset, conformal_dataset = random_split(
    full_train,
    [40000, 10000, 10000],
    generator=torch.Generator().manual_seed(42)
)


print("\n================ SPLITS ================")
print("Training:", len(train_dataset))
print("Tuning:", len(tuning_dataset))
print("Conformal calibration:", len(conformal_dataset))
print("Test:", len(test_dataset))


# ==========================================
# 3. LOOK AT ONE SAMPLE
# ==========================================

image, label = train_dataset[0]

print("\n================ SAMPLE ================")
print("Image shape:", image.shape)
print("Label:", label)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

print("Class:", class_names[label])

print("\nMinimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())

# ==========================================
# 4. CREATE DATA LOADERS
# ==========================================

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


# ==========================================
# 5. SMALL NEURAL NETWORK
# ==========================================

class FashionClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(

            nn.Flatten(),

            nn.Linear(28 * 28, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.network(x)


model = FashionClassifier()

print("\n================ MODEL ================")
print(model)


# ==========================================
# 6. TRAIN
# ==========================================

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 8

print("\n================ TRAINING ================")

for epoch in range(epochs):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        # Forward pass
        logits = model(images)

        loss = loss_function(logits, labels)

        # Backward pass
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics
        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1:2d} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ==========================================
# 7. TEST CLASSIFICATION ACCURACY
# ==========================================

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        logits = model(images)

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += labels.size(0)


test_accuracy = correct / total

print("\n================ TEST ================")
print(f"Test accuracy: {test_accuracy:.4f}")


# ==========================================
# 8. INSPECT ONE PREDICTION
# ==========================================

image, true_label = test_dataset[0]

with torch.no_grad():

    logits = model(image.unsqueeze(0))

    probabilities = torch.softmax(logits, dim=1)


print("\n================ EXAMPLE PREDICTION ================")

print("True class:")
print(true_label, class_names[true_label])

print("\nLogits:")
print(logits[0])

print("\nProbabilities:")

for i, probability in enumerate(probabilities[0]):

    print(
        f"{i}: {class_names[i]:12s} "
        f"{probability.item():.4f}"
    )

predicted_class = probabilities.argmax(dim=1).item()

print("\nPredicted class:")
print(predicted_class, class_names[predicted_class])

# ==========================================
# 9. COLLECT LOGITS FOR ALL DATASETS
# ==========================================

def collect_logits(model, loader):

    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            logits = model(images)

            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return all_logits, all_labels


tuning_logits, tuning_labels = collect_logits(
    model,
    tuning_loader
)

conformal_logits, conformal_labels = collect_logits(
    model,
    conformal_loader
)

test_logits, test_labels = collect_logits(
    model,
    test_loader
)


print("\n================ LOGIT SHAPES ================")

print(
    "Tuning:",
    tuning_logits.shape,
    tuning_labels.shape
)

print(
    "Conformal:",
    conformal_logits.shape,
    conformal_labels.shape
)

print(
    "Test:",
    test_logits.shape,
    test_labels.shape
)


# Optional: save them so we can reuse them later
torch.save(
    {
        "tuning_logits": tuning_logits,
        "tuning_labels": tuning_labels,

        "conformal_logits": conformal_logits,
        "conformal_labels": conformal_labels,

        "test_logits": test_logits,
        "test_labels": test_labels,
    },
    "fashion_mnist_logits.pt"
)

print("\nSaved logits to: fashion_mnist_logits.pt")