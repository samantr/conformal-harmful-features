import numpy as np


def softmax(logits):
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values)


logits = np.array([3.0, 1.5, 0.5])

temperatures = [0.5, 1.0, 2.0]

for T in temperatures:
    scaled_logits = logits / T
    probabilities = softmax(scaled_logits)

    print("Temperature:", T)
    print("Original logits:", logits)
    print("Scaled logits:", scaled_logits)
    print("Probabilities:", probabilities)
    print()