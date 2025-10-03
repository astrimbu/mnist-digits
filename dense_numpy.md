
How to build and train a dense neural network using **only NumPy**
(no Keras, no PyTorch)

---

## High-level Overview

For each **mini-batch** of data:

1. **Forward pass (prediction)**  
2. **Backward pass (learning)**  
3. **Parameter update (optimization)**  

---

## 1. Preprocessing and Data Handling

```python
# Load MNIST dataset (using Keras for data loading only)
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten images: (n_samples, 28, 28) -> (n_samples, 784)
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)
```

---

## 2. Core Architecture

### Network Structure

```
Input (784) → [Dense + ReLU] → Hidden (128) → [Dense + Softmax] → Output (10)
```

*784 pixels → hidden layer → 10 digit classes (0-9)*

### Creating the Network

```python
# Create model: 784 -> 128 -> 10
model = DenseNeuralNetwork(
    layer_sizes=[784, 128, 10],
    activations=['relu', 'softmax']
)
```

### Weight & Bias Initialization

```python
class DenseLayer:
    """A fully connected layer with weights, biases, and activation."""
    
    def __init__(self, input_size, output_size, activation='relu'):
        # Xavier/Glorot initialization: smart starting weights that help the network learn faster
        # (starts weights at a scale that prevents gradients from vanishing or exploding)
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        # Cache for backpropagation
        self.input = None
        self.z = None
        self.output = None
```

*Initialize weights with Xavier/Glorot initialization (smart starting values optimized for ReLU). Bias starts at 0.*  

### Full Network Structure

```python
class DenseNeuralNetwork:
    """A simple dense neural network."""
    
    def __init__(self, layer_sizes, activations):
        """
        Args:
            layer_sizes: [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = DenseLayer(layer_sizes[i], layer_sizes[i+1], activations[i])
            self.layers.append(layer)
```

---

## 3. Forward Pass

*Now that we've built the network structure, let's see how it makes predictions by passing data forward through each layer.*

---

### 3.1 Network-Level Forward Pass

```python
def forward(self, x):
    """Forward pass through the entire network."""
    output = x
    for layer in self.layers:
        output = layer.forward(output)
    return output
```

*Chain together all layers: output of one layer becomes input to the next.*

---

### 3.2 Layer-Level Linear Transformation

```python
def forward(self, x):
    """Forward pass through the layer."""
    self.input = x
    self.z = np.dot(x, self.weights) + self.biases
    # ... activation applied next
```

*Compute the raw weighted sum of inputs plus bias: `z = x @ weights + biases`*

- `x`: input to this layer. Shape: `(batch_size, input_size)`
- `self.weights`: weight matrix. Shape: `(input_size, output_size)`
- `self.biases`: bias vector. Shape: `(1, output_size)`
- `self.z`: raw pre-activation values. Shape: `(batch_size, output_size)`

---

### 3.3 Non-linear Activation (ReLU)

```python
# Inside forward method, after computing self.z
if self.activation == 'relu':
	self.output = np.maximum(0, self.z)
```

*Apply non-linearity. ReLU = max(0, z): passes positive values, zeros out negatives.*

ReLU introduces non-linearity, enabling stacked layers to approximate complex functions.  

---

### 3.4 Output Activation (Softmax)

```python
# Inside forward method
elif self.activation == 'softmax':
	# Numerically stable softmax
	exp_z = np.exp(self.z - np.max(self.z, axis=1, keepdims=True))
	self.output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

*Convert logits into a probability distribution (values between 0 and 1, sum to 1 per sample).*

- Subtract `max(self.z)` to prevent computer from trying to calculate impossibly large numbers (numerical stability)
- Output is a probability distribution perfect for multi-class classification

---

### 3.5 Loss Function: Cross-Entropy

*Now that we have predictions, we need to measure how wrong they are.*

```python
# Inside train_step method
# Convert labels to one-hot encoding
y_one_hot = np.zeros_like(predictions)
y_one_hot[np.arange(batch_size), y_batch] = 1

# Cross-entropy loss
loss = -np.sum(y_one_hot * np.log(predictions + 1e-8)) / batch_size
```

*Measures how different the predicted distribution is from the true distribution.*

- `y_batch`: integer labels, e.g., `[3, 7, 2, ...]`
- `y_one_hot`: one-hot encoded labels, e.g., `[[0,0,0,1,0,...], [0,0,0,0,0,0,0,1,0,0], ...]`
- `predictions`: softmax probabilities. Shape: `(batch_size, 10)`
- Add `1e-8` epsilon to avoid `log(0)` which would be undefined  

---

## 4. Backward Pass

*Now that we know what the network predicted (forward pass) and how wrong it was (loss), we work backwards to figure out which weights to adjust.*

---

### 4.1 Initial Error Signal (Softmax + Cross-Entropy)

```python
# Inside train_step method
# Backward pass (gradient of softmax + cross-entropy)
grad_output = predictions - y_one_hot
self.backward(grad_output, learning_rate)
```

*Gradient of loss with respect to the final layer's output.*

The elegant formula `predictions - y_one_hot` is the derivative of cross-entropy loss combined with softmax activation. If the prediction was `[0.1, 0.2, 0.7]` but true label was `[0, 0, 1]`, the gradient is `[0.1, 0.2, -0.3]` (penalizes the wrong predictions, rewards the correct one).  

---

### 4.2 Hidden Layer Gradient (ReLU Derivative)

```python
# Inside backward method
# Compute gradient with respect to the activation
if self.activation == 'relu':
	grad_z = grad_output * (self.z > 0)
```

*Apply ReLU derivative: gradient passes through if neuron was active (`z > 0`), otherwise blocked.*

- `(self.z > 0)`: boolean mask, `True` where neuron was active during forward pass
- Element-wise multiplication zeros out gradients for inactive neurons  

---

### 4.3 Gradients wrt Parameters

```python
# Inside backward method
def backward(self, grad_output, learning_rate):
	"""Backward pass through the layer."""
	# Apply activation derivative
	if self.activation == 'relu':
		grad_z = grad_output * (self.z > 0)
	elif self.activation == 'softmax':
		grad_z = grad_output  # Already simplified for softmax + cross-entropy
	
	# Compute gradients
	batch_size = self.input.shape[0]
	grad_weights = np.dot(self.input.T, grad_z) / batch_size
	grad_biases = np.sum(grad_z, axis=0, keepdims=True) / batch_size
	grad_input = np.dot(grad_z, self.weights.T)
	
	# Update weights and biases
	self.weights -= learning_rate * grad_weights
	self.biases -= learning_rate * grad_biases
	
	return grad_input
```

*Compute gradients for weights, biases, and propagate backward to previous layer.*

**Key calculations:**
- `grad_weights = self.input.T @ grad_z / batch_size`: How much each weight contributed to the error
- `grad_biases = np.sum(grad_z, ...) / batch_size`: How much each bias contributed to the error  
- `grad_input = grad_z @ self.weights.T`: Propagate error signal to previous layer

**Variables:**
- `self.input.T`: Transpose of input matrix (flip rows ↔ columns to align dimensions for matrix multiplication)
- `grad_z`: Gradient of loss with respect to this layer's pre-activation output
- `batch_size`: Number of samples, used to average gradients (so larger batches don't create artificially large updates)

---

### 4.4 Network-Level Backward Pass

```python
def backward(self, grad_output, learning_rate):
    """Backward pass through the entire network."""
    grad = grad_output
    for layer in reversed(self.layers):
        grad = layer.backward(grad, learning_rate)
```

*Propagate gradients backwards through all layers, starting from output layer.*

The gradient flows backward, layer by layer, until we've computed updates for all parameters in the network.

---

## 5. Optimizer & Training Loop

*With gradients computed, we can now update the parameters to improve the network.*

---

### 5.1 Update Rule (Stochastic Gradient Descent)

```python
# Inside backward method (already shown above)
# Update weights and biases
self.weights -= learning_rate * grad_weights
self.biases -= learning_rate * grad_biases
```

*Update parameters by stepping in the opposite direction of the gradient (toward lower loss).*

- `learning_rate`: Step size (typically 0.01 to 0.1)
- Negative sign: move parameters to reduce loss  

### 5.2 Training Loop

```python
def train_model(model, x_train, y_train, x_test, y_test, 
                epochs=5, batch_size=128, learning_rate=0.01):
    """Train the model for a given number of epochs."""
    n_samples = x_train.shape[0]
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(n_samples)
        x_train_shuffled = x_train[indices]
        y_train_shuffled = y_train[indices]
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            x_batch = x_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            loss, accuracy = model.train_step(x_batch, y_batch, learning_rate)
            # ... (continues with tracking and evaluation)
```

**The training process:**
1. Shuffle training data (prevents learning order-dependent patterns)
2. Split into mini-batches (more stable than single samples)
3. For each batch:  
   - Forward pass → compute loss → backward pass → update parameters

---

### 5.3 Training History Tracking

```python
history = {'train_loss': [], 'train_acc': [], 'test_acc': []}

# During training...
history['train_loss'].append(avg_loss)
history['train_acc'].append(avg_train_acc)
history['test_acc'].append(test_accuracy)
```

*We keep track of metrics over time to monitor learning progress and detect issues like overfitting.*

This history allows us to visualize how the network improves epoch by epoch.

---

## 6. Evaluation

```python
def predict(self, x):
    """Make predictions on input data."""
    output = self.forward(x)
    return np.argmax(output, axis=1)

def evaluate(self, x, y):
    """Evaluate the model on a dataset."""
    predictions = self.predict(x)
    accuracy = np.mean(predictions == y)
    return accuracy
```

**How it works:**
1. Forward pass on validation/test set
2. `np.argmax(output, axis=1)`: Convert probabilities to predicted class (highest probability wins)
3. `accuracy = np.mean(predictions == y)`: Percentage of correct predictions

Example: If `output = [[0.1, 0.2, 0.7]]`, then `argmax` returns `2` (the digit with highest probability).  

---

## 7. Running the Full Pipeline

### The `main()` Function

The `main()` function orchestrates the entire process:

```python
def main():
    # 1. Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0
    x_test = x_test.reshape(-1, 28*28) / 255.0
    
    # 2. Build the network architecture
    model = DenseNeuralNetwork(
        layer_sizes=[784, 128, 10],
        activations=['relu', 'softmax']
    )
    
    # 3. Train for 5 epochs
    history = train_model(
        model, x_train, y_train, x_test, y_test,
        epochs=5, batch_size=128, learning_rate=0.1
    )
    
    # 4. Visualize a random prediction alongside training curves
    # (Creates plots showing: prediction, loss curve, accuracy curve)
```

*This demonstrates the complete workflow from raw data to trained model with visualizations.*

### Visualization

The code creates three plots:
1. **Test Image**: Shows a random digit with predicted vs. actual label
2. **Training Loss**: Shows how loss decreases over epochs
3. **Accuracy Curves**: Compares training vs. test accuracy to detect overfitting

---

## Summary of Flow

1. Preprocess inputs (normalize, flatten)
2. Initialize weights and biases (Xavier initialization)
3. Forward pass: compute `z`, apply activation, predict with softmax
4. Compute cross-entropy loss
5. Backward pass: compute gradients (`grad_weights`, `grad_biases`, `grad_input`)
6. Update parameters with SGD
7. Track metrics and evaluate on test data
8. Visualize results

---

## Complete Example Usage

```python
# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test = x_test.reshape(-1, 28*28) / 255.0

# 2. Create the network
model = DenseNeuralNetwork(
    layer_sizes=[784, 128, 10],
    activations=['relu', 'softmax']
)

# 3. Train the model
history = train_model(
    model, x_train, y_train, x_test, y_test,
    epochs=5, batch_size=128, learning_rate=0.1
)

# 4. Evaluate
final_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {final_accuracy:.4f}")

# 5. Make predictions
predictions = model.predict(x_test[:10])
```

---

## Quick Reference / Glossary

**Core Concepts:**
- **Batch**: A small subset of training data processed together (e.g., 128 samples)
- **Epoch**: One complete pass through the entire training dataset
- **Forward Pass**: Computing predictions by passing data through the network
- **Backward Pass**: Computing gradients by working backwards from the error

**Mathematical Terms:**
- **Gradient**: Directional derivative showing how to adjust a parameter to reduce loss
- **Learning Rate**: Step size for parameter updates (too large = unstable, too small = slow)
- **Loss**: A number measuring how wrong the predictions are (lower is better)
- **Softmax**: Converts raw scores into probabilities that sum to 1

**Matrix Operations:**
- **Transpose (`.T`)**: Flip rows and columns of a matrix
- **Dot Product (`@` or `np.dot`)**: Matrix multiplication
- **Element-wise Multiplication (`*`)**: Multiply corresponding elements

**Layers & Activations:**
- **Dense Layer**: Fully connected layer where every input connects to every output
- **ReLU**: Rectified Linear Unit, zeros out negative values: `max(0, x)`
- **Xavier Initialization**: Smart starting weights that help networks learn faster  
