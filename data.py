"""
Explores the structure and content of the MNIST dataset.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training data (x_train):")
print(f"  - Shape: {x_train.shape}")
print(f"  - Data type: {x_train.dtype}")
print(f"  - Memory usage: {x_train.nbytes / (1024**2):.2f} MB")

print(f"\nTraining labels (y_train):")
print(f"  - Shape: {y_train.shape}")
print(f"  - Data type: {y_train.dtype}")
print(f"  - Unique values: {np.unique(y_train)}")
print(f"  - Label distribution: {np.bincount(y_train)}")

print(f"\nTest data (x_test):")
print(f"  - Shape: {x_test.shape}")
print(f"  - Data type: {x_test.dtype}")
print(f"  - Memory usage: {x_test.nbytes / (1024**2):.2f} MB")

print(f"\nTest labels (y_test):")
print(f"  - Shape: {y_test.shape}")
print(f"  - Data type: {y_test.dtype}")
print(f"  - Unique values: {np.unique(y_test)}")
print(f"  - Label distribution: {np.bincount(y_test)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Sample Images from Each Digit Class (0-9)', fontsize=16)

for digit in range(10):
    # Find first occurrence of each digit in training data
    idx = np.where(y_train == digit)[0][0]
    
    row = digit // 5
    col = digit % 5
    
    axes[row, col].imshow(x_train[idx], cmap='gray')
    axes[row, col].set_title(f'Digit: {digit}\nIndex: {idx}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
