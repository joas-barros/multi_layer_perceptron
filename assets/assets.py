import numpy as np
from util.activation_functions import *
from util.derivatives import *
from util.loss_functions import *

# Entradas normalizadas
X_train = np.array([
    [0.0],
    [0.14],
    [0.28],
    [0.42],
    [0.57],
    [0.71],
    [0.85],
    [1.0]
])

# Saídas desejadas (binário)
Y_train = np.array([
    [0, 0, 0],  # 0
    [0, 0, 1],  # 1
    [0, 1, 0],  # 2
    [0, 1, 1],  # 3
    [1, 0, 0],  # 4
    [1, 0, 1],  # 5
    [1, 1, 0],  # 6
    [1, 1, 1]   # 7
])

input_size = 1
output_size = 3
hidden_size = 8
learning_rate = 0.5
epochs = 10000

activation_function = tanh  # 'sigmoid', 'relu', 'tanh', 'softmax'
loss_function = mean_squared_error
weight_init = 'xavier'  # 'xavier' or 'random'
