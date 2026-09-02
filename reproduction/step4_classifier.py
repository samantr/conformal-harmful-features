from sklearn.datasets import make_classification
import torch
import torch.nn as nn


# -------------------------------------------------
# 1. Create the same dataset from Step 2
# -------------------------------------------------

X, y = make_classification(
    n_samples=6000,
    n_features=10,
    n_informative=8,
    n_redundant=0,
    n_classes=5,
    n_clusters_per_class=1,
    random_state=42
)


# -------------------------------------------------
# 2. Convert NumPy data to PyTorch tensors
# -------------------------------------------------

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)


# -------------------------------------------------
# 3. Make results reproducible
# -------------------------------------------------

torch.manual_seed(42)


# -------------------------------------------------
# 4. Create our tiny neural network
# -------------------------------------------------

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),

    nn.Linear(32, 32),
    nn.ReLU(),

    nn.Linear(32, 5)
)


# -------------------------------------------------
# 5. Training tools
# -------------------------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# -------------------------------------------------
# 6. Train
# -------------------------------------------------

epochs = 200

for epoch in range(epochs):

    # Network produces logits
    logits = model(X)

    # Compare logits with true classes
    loss = loss_function(logits, y)

    # Clear old gradients
    optimizer.zero_grad()

    # Calculate new gradients
    loss.backward()

    # Update neural-network weights
    optimizer.step()

    if (epoch + 1) % 20 == 0:

        predictions = logits.argmax(dim=1)

        accuracy = (
            predictions == y
        ).float().mean()

        print(
            f"Epoch {epoch + 1:3d} | "
            f"Loss: {loss.item():.4f} | "
            f"Accuracy: {accuracy.item():.4f}"
        )


# -------------------------------------------------
# 7. Inspect the trained model
# -------------------------------------------------

model.eval()

with torch.no_grad():

    logits = model(X)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = logits.argmax(dim=1)


# -------------------------------------------------
# 8. Look at the first 5 samples
# -------------------------------------------------

print("\nFirst 5 true classes:")
print(y[:5])

print("\nFirst 5 predicted classes:")
print(predictions[:5])

print("\nFirst 5 logits:")
print(logits[:5])

print("\nFirst 5 probabilities:")
print(probabilities[:5])

print("\nProbability sums:")
print(probabilities[:5].sum(dim=1))