from util.activation_functions import *

def sigmoid_derivative(x):
    """
    Compute the derivative of the sigmoid function.
    
    Parameters:
    x (float or np.ndarray): Input value(s) for which to compute the derivative.
    
    Returns:
    float or np.ndarray: The derivative of the sigmoid function evaluated at x.
    """
    sig = sigmoid(x)
    return sig * (1 - sig)

def relu_derivative(x):
    """
    Compute the derivative of the ReLU function.
    
    Parameters:
    x (float or np.ndarray): Input value(s) for which to compute the derivative.
    
    Returns:
    float or np.ndarray: The derivative of the ReLU function evaluated at x.
    """
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    """
    Compute the derivative of the tanh function.
    
    Parameters:
    x (float or np.ndarray): Input value(s) for which to compute the derivative.
    
    Returns:
    float or np.ndarray: The derivative of the tanh function evaluated at x.
    """
    return 1 - tanh(x) ** 2

def softmax_derivative(x):
    """
    Compute the derivative of the softmax function.
    
    Parameters:
    x (float or np.ndarray): Input value(s) for which to compute the derivative.
    
    Returns:
    np.ndarray: The Jacobian matrix of the softmax function evaluated at x.
    """
    s = softmax(x)
    return np.diagflat(s) - np.outer(s, s)