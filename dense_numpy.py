"""
Implements a dense neural network from scratch using only NumPy.
Uses Keras only for loading the MNIST dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.datasets import mnist


class DenseLayer:
    """A fully connected layer with weights, biases, and activation."""
    
    def __init__(self, input_size, output_size, activation='relu'):
        # Xavier/Glorot initialization for better convergence
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # Cache for backpropagation
        self.input = None
        self.z = None
        self.output = None
        
    def forward(self, x):
        """Forward pass through the layer."""
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.z)
        elif self.activation == 'softmax':
            # Numerically stable softmax
            exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
            self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif self.activation == 'none':
            self.output = self.z
        
        return self.output
    
    def backward(self, grad_output, learning_rate):
        """Backward pass through the layer."""
        # Compute gradient with respect to the activation
        if self.activation == 'relu':
            grad_z = grad_output * (self.z > 0)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, the gradient is already simplified
            grad_z = grad_output
        elif self.activation == 'none':
            grad_z = grad_output
        
        # Compute gradients
        batch_size = self.input.shape[0]
        grad_weights = np.dot(self.input.T, grad_z) / batch_size
        grad_biases = np.sum(grad_z, axis=0, keepdims=True) / batch_size
        grad_input = np.dot(grad_z, self.weights.T)
        
        # Update weights and biases
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input


class DenseNeuralNetwork:
    """A simple dense neural network with multiple layers."""
    
    def __init__(self, layer_sizes, activations):
        """
        Initialize the network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer (excluding input)
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
    
    def forward(self, x):
        """Forward pass through the entire network."""
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def backward(self, grad_output, learning_rate):
        """Backward pass through the entire network."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
    
    def train_step(self, x_batch, y_batch, learning_rate):
        """Perform one training step (forward + backward)."""
        # Forward pass
        predictions = self.forward(x_batch)
        
        # Compute loss (cross-entropy)
        batch_size = x_batch.shape[0]
        
        # Convert labels to one-hot encoding
        y_one_hot = np.zeros_like(predictions)
        y_one_hot[np.arange(batch_size), y_batch] = 1
        
        # Cross-entropy loss
        loss = -np.sum(y_one_hot * np.log(predictions + 1e-8)) / batch_size
        
        # Accuracy
        predictions_class = np.argmax(predictions, axis=1)
        accuracy = np.mean(predictions_class == y_batch)
        
        # Backward pass (gradient of softmax + cross-entropy)
        grad_output = predictions - y_one_hot
        self.backward(grad_output, learning_rate)
        
        return loss, accuracy
    
    def predict(self, x):
        """Make predictions on input data."""
        output = self.forward(x)
        return np.argmax(output, axis=1)
    
    def evaluate(self, x, y):
        """Evaluate the model on a dataset."""
        predictions = self.predict(x)
        accuracy = np.mean(predictions == y)
        return accuracy


def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=5, batch_size=128, learning_rate=0.01):
    """Train the model for a given number of epochs."""
    n_samples = x_train.shape[0]
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        epoch_losses = []
        epoch_accuracies = []
        
        for i in range(0, n_samples, batch_size):
            x_batch = x_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            loss, accuracy = model.train_step(x_batch, y_batch, learning_rate)
            epoch_losses.append(loss)
            epoch_accuracies.append(accuracy)
        
        # Evaluate on test set
        test_accuracy = model.evaluate(x_test, y_test)
        
        avg_loss = np.mean(epoch_losses)
        avg_train_acc = np.mean(epoch_accuracies)
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_train_acc)
        history['test_acc'].append(test_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f} - "
              f"Train Acc: {avg_train_acc:.4f} - "
              f"Test Acc: {test_accuracy:.4f}")
    
    return history


def main():
    """Main function to train and evaluate the model."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Flatten images
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    
    # Create model: 784 -> 128 -> 10 (similar to dense.py)
    print("\nBuilding neural network from scratch...")
    model = DenseNeuralNetwork(
        layer_sizes=[784, 128, 10],
        activations=['relu', 'softmax']
    )
    
    # Train the model
    print("\nTraining...")
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=5, batch_size=128, learning_rate=0.1
    )
    
    # Final evaluation
    final_accuracy = model.evaluate(x_test, y_test)
    print(f"\nFinal Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    # Test on a random sample
    test_index = random.randint(0, len(x_test) - 1)
    test_sample = x_test[test_index].reshape(1, -1)
    predicted_class = model.predict(test_sample)[0]
    actual_class = y_test[test_index]
    
    # Visualize prediction
    plt.figure(figsize=(12, 4))
    
    # Show the test image
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[test_index].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted: {predicted_class} | Actual: {actual_class}")
    plt.axis('off')
    
    # Plot training history
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

