"""
Trains a neural network on the MNIST dataset and tests it against a custom image.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import datetime
import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

model = tf.keras.Sequential([
    Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
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

save_model = input("\nSave the trained neural network? (y/n): ").lower().strip()

if save_model == 'y' or save_model == 'yes':
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_filename = f"mnist_model_{timestamp}.keras"

    try:
        model.save(model_filename)
        print(f"Model saved successfully as '{model_filename}'")
    except Exception as e:
        print(f"Error saving model: {e}")
