"""
Trains a neural network on the MNIST dataset and tests it against a custom image.
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=3,
    validation_data=(x_test, y_test)
)

test_index = random.randint(0, len(x_test))
test_sample = x_test[test_index].reshape(1, -1)
probabilities = model.predict(test_sample)
predicted_class = probabilities.argmax()
actual_class = y_test[test_index]

plt.imshow(x_test[test_index].reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_class} | Actual: {actual_class}")
plt.show()

img = Image.open('test.png').convert('L')
img_array = np.array(img)

if img_array.mean() > 127:
    img_array = 255 - img_array

img_array = img_array / 255.0
img_flat = img_array.reshape(1, -1)
prediction = model.predict(img_flat)

top3_indices = prediction[0].argsort()[-3:][::-1]
print("\nTop 3 predictions for test.png:")
for i, idx in enumerate(top3_indices):
    confidence = prediction[0][idx] * 100
    print(f"{i+1}. Digit {idx}: {confidence:.1f}%")
    
predicted_digit = prediction.argmax()

plt.imshow(img_array, cmap='gray')
plt.title(f"Predicted: {predicted_digit}")
plt.show()
