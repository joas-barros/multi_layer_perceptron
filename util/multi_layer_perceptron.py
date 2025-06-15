import numpy as np
from util.activation_functions import *
from util.derivatives import *
from util.loss_functions import *

class MultiLayerPerceptron:

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, 
                 activation_function=sigmoid, 
                 loss_function=mean_squared_error,
                  weight_init='random'):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.loss_function = loss_function
        self.weight_init = weight_init

        self._set_derivative_loss_function()
        self._set_derivative_activation_function()

        # Inicializando os pesos e vieses
        self._initialize_weights()
    
    def _initialize_weights(self):
        if self.weight_init == 'random':
            self.weights_input_hidden = np.random.uniform(-1, 1, (self.input_size, self.hidden_size))
            self.weights_hidden_output = np.random.uniform(-1, 1, (self.hidden_size, self.output_size))
            self.bias_hidden = np.random.uniform(-1, 1, (1, self.hidden_size))
            self.bias_output = np.random.uniform(-1, 1, (1, self.output_size))

        elif self.weight_init == 'xavier':
            limit_input_hidden = np.sqrt(6 / (self.input_size + self.hidden_size))
            limit_hidden_output = np.sqrt(6 / (self.hidden_size + self.output_size))
            self.weights_input_hidden = np.random.uniform(-limit_input_hidden, limit_input_hidden, 
                                                          (self.input_size, self.hidden_size))
            self.weights_hidden_output = np.random.uniform(-limit_hidden_output, limit_hidden_output, 
                                                           (self.hidden_size, self.output_size))
            self.bias_hidden = np.zeros((1, self.hidden_size))
            self.bias_output = np.zeros((1, self.output_size))
        else:
            raise ValueError("Invalid weight initialization method. Use 'random' or 'xavier'.")
    
    def _set_derivative_loss_function(self):
        loss_derivative_dict = {
            mean_squared_error: mean_squared_error_derivative,
            binary_cross_entropy: binary_cross_entropy_derivative
        }

        if self.loss_function in loss_derivative_dict:
            self.loss_derivative = loss_derivative_dict[self.loss_function]
        else:
            raise ValueError("Unsupported loss function. Use mean_squared_error or binary_cross_entropy.")
    
    def _set_derivative_activation_function(self):
        derivative_dict = {
            sigmoid: sigmoid_derivative,
            relu: relu_derivative,
            tanh: tanh_derivative,
            softmax: softmax_derivative
        }

        if self.activation_function in derivative_dict:
            self.derivative_function = derivative_dict[self.activation_function]
        else:
            raise ValueError("Unsupported activation function. Use sigmoid, relu, tanh, or softmax.")


    def _feedforward(self, inputs):

        # 1. Calculando a entrada ponderada para a camada oculta (Z_oculta)
        self.hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden

        # 2. Aplicando a função de ativação na camada oculta (A_oculta)
        self.hidden_output = self.activation_function(self.hidden_input)

        # 3. Calculando a entrada ponderada para a camada de saída (Z_saida)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output

        # 4. Aplicando a função de ativação na camada de saída (A_saida)
        self.output = sigmoid(self.output_input) # Sigmoid sempre será a função de ativação da camada de saída pela natureza do problema ser binaria

        return self.output
    
    def _backpropagation(self, inputs, targets):
        # 1. Calculando o erro na camada de saída
        output_error = self.loss_derivative(targets, self.output)

        # 2. Calculando o delta (gradiente local) da camada de saída
        output_delta = output_error * sigmoid_derivative(self.output_input) # Sigmoid sempre será a função de ativação da camada de saída pela natureza do problema ser binaria

        # 3. Calculando o erro na camada oculta (retropropagando o erro da saída)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)

        # 4. Calculando o delta (gradiente local) da camada oculta
        hidden_delta = hidden_error * self.derivative_function(self.hidden_input)

        # Calculando os gradientes para atualização
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_delta)
        d_bias_output = np.sum(output_delta, axis=0, keepdims=True)
        d_weights_input_hidden = np.dot(inputs.T, hidden_delta)
        d_bias_hidden = np.sum(hidden_delta, axis=0, keepdims=True)

        return (d_weights_input_hidden, d_bias_hidden, 
                d_weights_hidden_output, d_bias_output)

    def _update_weights(self, d_weights_input_hidden, d_bias_hidden,
                        d_weights_hidden_output, d_bias_output):
        
        self.weights_input_hidden += self.learning_rate * d_weights_input_hidden
        self.bias_hidden += self.learning_rate * d_bias_hidden
        self.weights_hidden_output += self.learning_rate * d_weights_hidden_output
        self.bias_output += self.learning_rate * d_bias_output
    
    def train(self, X_train, y_train, epochs):
        print("Training started...")
        print("=" * 30)
        for epoch in range(epochs):
            #feedforward operation
            output = self._feedforward(X_train)

            #backpropagation operation
            d_weights_input_hidden, d_bias_hidden, d_weights_hidden_output, d_bias_output = self._backpropagation(X_train, y_train)

            #update weights and biases
            self._update_weights(d_weights_input_hidden, d_bias_hidden, 
                                 d_weights_hidden_output, d_bias_output)
            
            if epoch % 100 == 0:
                loss = self.loss_function(y_train, output)
                accuracy = accuracy_function(y_train, output) * 100  # Convert to percentage
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Accuracy: {accuracy:.2f} %")
        
        print("=" * 30)
        print("Training completed.")

    def predict(self, X):
        """Make predictions for the input data."""
        output = self._feedforward(X)
        return np.round(output)

