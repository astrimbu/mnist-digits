"""
Quick exploration of the MNIST dataset structure and sample images.
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Training set: {x_train.shape} images, {x_train.nbytes / (1024**2):.1f} MB")
print(f"Test set: {x_test.shape} images, {x_test.nbytes / (1024**2):.1f} MB")
print(f"Labels: {np.unique(y_train)} (digits 0-9)")
print(f"Training samples per digit: {np.bincount(y_train)}")

fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Sample Images for Each Digit (0-9)', fontsize=16)

for digit in range(10):
    # Find first occurrence of each digit in training data
    idx = np.where(y_train == digit)[0][0]
    
    row = digit // 5
    col = digit % 5
    
    axes[row, col].imshow(x_train[idx], cmap='gray')
    axes[row, col].set_title(f'Digit {digit}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()
