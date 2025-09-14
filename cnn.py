"""
Trains a convolutional neural network on the MNIST dataset.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.models import Sequential

tf.get_logger().setLevel('ERROR')

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape to keep spatial structure (28, 28, 1) instead of flattening
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Create CNN model
model = Sequential([
    # Input layer to define the shape
    Input(shape=(28, 28, 1)),
    
    # First conv layer: 32 filters to detect basic features
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Second conv layer: 64 filters for more complex patterns
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten and classify
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 digits (0-9)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("CNN Architecture:")
model.summary()

print("\nTraining...")
history = model.fit(
    x_train, y_train,
    epochs=3,
    validation_data=(x_test, y_test),
    verbose=1
)

# Test on random sample from test set
test_index = random.randint(0, len(x_test) - 1)
test_sample = x_test[test_index:test_index+1]  # Keep batch dimension
probabilities = model.predict(test_sample)
predicted_class = probabilities.argmax()
actual_class = y_test[test_index]

plt.imshow(x_test[test_index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_class} | Actual: {actual_class}")
plt.show()

print(f"\nResults:")
print(f"Training: {history.history['accuracy'][-1]:.4f}")
print(f"Validation: {history.history['val_accuracy'][-1]:.4f}")
