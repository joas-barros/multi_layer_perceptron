from util.multi_layer_perceptron import MultiLayerPerceptron
from assets.assets import *

mlp = MultiLayerPerceptron(input_size, hidden_size, output_size, learning_rate, 
                           activation_function, derivative_function)

# Treinamento do MLP
mlp.train(X_train, Y_train, epochs)


# Testando o MLP
print("Resultados do treinamento:")
for i in range(len(X_train)):
    prediction = mlp.predict(X_train[i])
    print(f"Entrada: {X_train[i][0]:.2f}, Previsão: {prediction}, Esperado: {Y_train[i].tolist()}")