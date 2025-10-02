# MNIST Digit Recognition

A machine learning project exploring different neural network architectures for handwritten digit classification using the MNIST dataset.

## Neural Network Implementations

This project compares different approaches to digit classification:

### Dense Neural Network (`dense.py`)
- Fully-connected layers with flattened input (using Keras)
- Architecture: Input(784) → Dense(128) → Dense(10)
- Accuracy: ~95-98%

### Dense Neural Network from Scratch (`dense_numpy.py`)
- Fully-connected layers implemented from scratch with NumPy
- Implements forward propagation, backpropagation, and gradient descent
- Architecture: Input(784) → Dense(128) → Dense(10)
- Accuracy: ~95-97%

### Basic CNN (`cnn.py`)
- Simple convolutional neural network
- Preserves spatial structure of images
- Accuracy: ~99%+

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Explore the dataset
python data.py

# Try each model
python dense.py           # Keras implementation
python dense_numpy.py     # NumPy from scratch
python cnn.py
```

## Requirements

- TensorFlow
- NumPy
- Matplotlib
- PIL (Pillow)

## Why CNNs Work Better for Images

CNNs outperform dense networks on image tasks because they:
- Understand spatial relationships between pixels
- Share parameters across the image (reducing overfitting)
- Build hierarchical features (edges → shapes → objects)
- Handle translation and small position changes well

The 4-6% accuracy jump from dense networks to CNNs shows how much architecture matters for your data type.
