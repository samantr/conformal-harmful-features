from sklearn.model_selection import train_test_split
import numpy as np

from step2_dataset import X, y

# ------------------------------------------------
# Split 1:
# 3000 training
# 3000 remaining
# ------------------------------------------------

X_train, X_remaining, y_train, y_remaining = train_test_split(
    X,
    y,
    train_size=3000,
    random_state=42,
    stratify=y
)


# ------------------------------------------------
# Split 2:
# From the remaining 3000:
# 1000 tuning
# 2000 remaining
# ------------------------------------------------

X_tune, X_remaining, y_tune, y_remaining = train_test_split(
    X_remaining,
    y_remaining,
    train_size=1000,
    random_state=42,
    stratify=y_remaining
)


# ------------------------------------------------
# Split 3:
# Divide the last 2000 into:
# 1000 conformal calibration
# 1000 test
# ------------------------------------------------

X_conf, X_test, y_conf, y_test = train_test_split(
    X_remaining,
    y_remaining,
    train_size=1000,
    random_state=42,
    stratify=y_remaining
)


print("Training:")
print(X_train.shape, y_train.shape)
print("Class counts:", np.bincount(y_train))

print("\nTuning:")
print(X_tune.shape, y_tune.shape)
print("Class counts:", np.bincount(y_tune))

print("\nConformal calibration:")
print(X_conf.shape, y_conf.shape)
print("Class counts:", np.bincount(y_conf))

print("\nTest:")
print(X_test.shape, y_test.shape)
print("Class counts:", np.bincount(y_test))