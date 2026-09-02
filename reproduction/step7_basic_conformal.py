import numpy as np
import torch
import torch.nn as nn

from step5_splits import (
    X_train,
    y_train,
    X_conf,
    y_conf,
    X_test,
    y_test
)

from collections import Counter



# -------------------------------------------------
# 1. Re-create the same neural network from Step 4
# -------------------------------------------------

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),

    nn.Linear(32, 32),
    nn.ReLU(),

    nn.Linear(32, 5)
)


# -------------------------------------------------
# 2. Convert TRAINING data to PyTorch tensors
# -------------------------------------------------

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)


# -------------------------------------------------
# 3. Train the model
# -------------------------------------------------

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

epochs = 200

for epoch in range(epochs):

    logits = model(X_train_tensor)

    loss = loss_function(
        logits,
        y_train_tensor
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


print("\nModel training finished.")


# -------------------------------------------------
# 4. Function:
#    X -> logits -> probabilities
# -------------------------------------------------

def get_probabilities(model, X):

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        logits = model(X_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    return probabilities.numpy()


# -------------------------------------------------
# 5. Get probabilities for CONFORMAL data
# -------------------------------------------------

conformal_probs = get_probabilities(
    model,
    X_conf
)


# -------------------------------------------------
# 6. Calculate TRUE-class conformal scores
#
#    score = 1 - probability of true class
# -------------------------------------------------

n = len(y_conf)

true_class_probs = conformal_probs[
    np.arange(n),
    y_conf
]

conformal_scores = (
    1.0 - true_class_probs
)


# -------------------------------------------------
# 7. Calculate conformal threshold
# -------------------------------------------------

alpha = 0.1

k = int(
    np.ceil(
        (n + 1) * (1 - alpha)
    )
)

sorted_scores = np.sort(
    conformal_scores
)

threshold = sorted_scores[k - 1]


print("\nNumber of conformal samples:", n)
print("Alpha:", alpha)
print("Target coverage:", 1 - alpha)
print("k:", k)
print("Conformal threshold:", threshold)


# -------------------------------------------------
# 8. Get probabilities for TEST data
# -------------------------------------------------

test_probs = get_probabilities(
    model,
    X_test
)


# -------------------------------------------------
# 9. Construct prediction sets
# -------------------------------------------------

prediction_sets = []

for probs in test_probs:

    candidate_scores = (
        1.0 - probs
    )

    prediction_set = np.where(
        candidate_scores <= threshold
    )[0]

    prediction_sets.append(
        prediction_set
    )


# -------------------------------------------------
# 10. Inspect first 5 test examples
# -------------------------------------------------

for i in range(5):

    print("\n----------------------------")
    print("Test sample:", i)

    print(
        "True class:",
        y_test[i]
    )

    print("\nProbabilities:")

    for class_id, probability in enumerate(
        test_probs[i]
    ):

        print(
            f"Class {class_id}: "
            f"{probability:.4f}"
        )


    print("\nCandidate scores:")

    for class_id, score in enumerate(
        1.0 - test_probs[i]
    ):

        print(
            f"Class {class_id}: "
            f"{score:.4f}"
        )


    print(
        "\nPrediction set:",
        prediction_sets[i].tolist()
    )

    print(
        "True class included:",
        y_test[i] in prediction_sets[i]
    )

print("\nFIRST TEST EXAMPLE WHERE TRUE CLASS IS NOT INCLUDED:")

for i in range(len(y_test)):

    if y_test[i] not in prediction_sets[i]:

        print("Test sample:", i)
        print("True class:", y_test[i])

        print("\nProbabilities:")
        for class_id, probability in enumerate(test_probs[i]):
            print(
                f"Class {class_id}: {probability:.4f}"
            )

        print(
            "\nPrediction set:",
            prediction_sets[i].tolist()
        )

        break

# -----------------------------
# Coverage
# -----------------------------

correct = 0

for true_label, pred_set in zip(y_test, prediction_sets):
    if true_label in pred_set:
        correct += 1

coverage = correct / len(y_test)

print("\nCoverage:")
print(coverage)


# -----------------------------
# Average prediction-set size
# -----------------------------

total_size = 0

for pred_set in prediction_sets:
    total_size += len(pred_set)

average_size = total_size / len(prediction_sets)

print("\nAverage prediction-set size:")
print(average_size)

set_sizes = [len(pred_set) for pred_set in prediction_sets]

print("\nPrediction-set size counts:")
print(Counter(set_sizes))