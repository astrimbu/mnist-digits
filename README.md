# MNIST Digit Recognition

A machine learning project exploring different neural network architectures for handwritten digit classification using the MNIST dataset.

## Neural Network Implementations

This project compares two approaches to digit classification:

### Dense Neural Network (`dense.py`)
- Fully-connected layers with flattened input
- Architecture: Input(784) → Dense(128) → Dense(10)
- Accuracy: ~95-98%
- Good starting point for understanding basic neural networks

### Basic CNN (`cnn.py`)
- Simple convolutional neural network
- Preserves spatial structure of images
- Accuracy: ~99%+
- Shows the power of CNNs for image tasks

## Quick Start

```bash
# Explore the dataset
python data.py

# Try each model
python dense.py
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
