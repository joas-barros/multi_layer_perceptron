import numpy as np

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Tanh activation function."""
    return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))

def softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0, keepdims=True)