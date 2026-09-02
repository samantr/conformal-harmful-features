import numpy as np

confidences = np.array([
    0.90, 0.85, 0.80,
    0.60, 0.55, 0.50
])

correct = np.array([
    1, 1, 0,
    1, 0, 0
])

bin_edges = np.array([0.0, 0.7, 1.0])

ece = 0.0

for i in range(len(bin_edges) - 1):

    lower = bin_edges[i]
    upper = bin_edges[i + 1]

    if i == 0:
        in_bin = (confidences >= lower) & (confidences <= upper)
    else:
        in_bin = (confidences > lower) & (confidences <= upper)

    count = np.sum(in_bin)

    if count == 0:
        continue

    bin_confidence = np.mean(confidences[in_bin])
    bin_accuracy = np.mean(correct[in_bin])

    difference = abs(bin_accuracy - bin_confidence)

    weight = count / len(confidences)

    ece += weight * difference

    print(f"Bin {i + 1}")
    print(f"Samples: {count}")
    print(f"Average confidence: {bin_confidence:.3f}")
    print(f"Accuracy: {bin_accuracy:.3f}")
    print(f"Difference: {difference:.3f}")
    print()

print(f"ECE: {ece:.3f}")