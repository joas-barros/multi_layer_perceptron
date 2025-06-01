class MultiLayerPerceptron:
    def feedforward(self, inputs):
        # 1. Calculando a entrada ponderada para a camada oculta (Z_oculta)

        # 2. Aplicando a função de ativação na camada oculta (A_oculta)

        # 3. Calculando a entrada ponderada para a camada de saída (Z_saida)

        # 4. Aplicando a função de ativação na camada de saída (A_saida)
        pass
    
    def backpropagation(self, inputs, targets):
        # 1. Calculando o erro na camada de saída

        # 2. Calculando o delta (gradiente local) da camada de saída

        # 3. Calculando o erro na camada oculta (retropropagando o erro da saída)

        # 4. Calculando o delta (gradiente local) da camada oculta

        # Calculando os gradientes para atualização
        pass

    def update_weights(self, learning_rate):
         """
        Atualiza os pesos e vieses da rede usando os gradientes calculados.

        Args:
            d_pesos_oculta_saida (np.array): Gradiente dos pesos da camada oculta para a saída.
            d_vieses_saida (np.array): Gradiente dos vieses da camada de saída.
            d_pesos_entrada_oculta (np.array): Gradiente dos pesos da entrada para a camada oculta.
            d_vieses_oculta (np.array): Gradiente dos vieses da camada oculta.
        """
         pass
    
    def train(self, X_train, y_train, epochs):
        pass

    def predict(self, X):
        pass

