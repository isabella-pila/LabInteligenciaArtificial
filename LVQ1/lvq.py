import numpy as np

class LVQ1:
    def __init__(self, n_features, n_classes, alpha=0.05, decay_rate=0.99):
        self.n_features = n_features
        self.n_classes = n_classes
        self.alpha = alpha
        self.decay_rate = decay_rate
        # Reference vectors (weights) and their corresponding classes
        self.w = None
        self.w_classes = None

    def initialize_weights(self, X, y):
        # Standard initialization: take the first pattern of each class as reference vector
        self.w = []
        self.w_classes = []
        for c in range(1, self.n_classes + 1):
            indices = np.where(y == c)[0]
            if len(indices) > 0:
                first_idx = indices[0]
                self.w.append(X[first_idx].copy())
                self.w_classes.append(c)
        self.w = np.array(self.w)
        self.w_classes = np.array(self.w_classes)

    def find_winner(self, x):
        # Euclidean distance
        distances = np.sum((self.w - x) ** 2, axis=1)
        winner_idx = np.argmin(distances)
        return winner_idx, distances[winner_idx]

    def train_pattern(self, x, target_class):
        winner_idx, _ = self.find_winner(x)
        winner_class = self.w_classes[winner_idx]
        
        # Weight update rule
        if winner_class == target_class:
            self.w[winner_idx] += self.alpha * (x - self.w[winner_idx])
        else:
            self.w[winner_idx] -= self.alpha * (x - self.w[winner_idx])
        return winner_class

    def train(self, X, y, epochs=100):
        # If weights are not initialized, initialize them
        if self.w is None:
            self.initialize_weights(X, y)
            
        initial_alpha = self.alpha
        for epoch in range(epochs):
            # Present each pattern in order
            for i in range(len(X)):
                self.train_pattern(X[i], y[i])
            # Decay learning rate linearly
            self.alpha = initial_alpha * (1.0 - (epoch / epochs))

    def predict(self, X):
        predictions = []
        for x in X:
            winner_idx, _ = self.find_winner(x)
            predictions.append(self.w_classes[winner_idx])
        return np.array(predictions)

# Training data
X_train = np.array([
    [2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649], # Amostra 1 (Classe 1)
    [2.3936, 1.4804, 1.9907, 1.2732, 2.2719, 1.8110], # Amostra 2 (Classe 1)
    [2.2880, 1.4585, 1.9867, 1.2451, 2.3389, 1.8099], # Amostra 3 (Classe 1)
    [2.2904, 1.4766, 1.8876, 1.2706, 2.2966, 1.7744], # Amostra 4 (Classe 1)
    [1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276], # Amostra 5 (Classe 2)
    [0.9913, 0.1524, 1.2700, 5.3808, 3.0714, 2.3331], # Amostra 6 (Classe 2)
    [1.0915, 0.1881, 1.1387, 5.3701, 3.2561, 2.3383], # Amostra 7 (Classe 2)
    [1.0535, 0.1229, 1.2743, 5.3226, 3.0950, 2.3193], # Amostra 8 (Classe 2)
    [1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850], # Amostra 9 (Classe 3)
    [1.3312, 2.2553, 0.9618, 2.4702, 1.7272, 5.0645], # Amostra 10 (Classe 3)
    [1.3646, 2.2945, 1.0562, 2.4763, 1.8051, 5.1470], # Amostra 11 (Classe 3)
    [1.4392, 2.2296, 1.1278, 2.4230, 1.7259, 5.0876], # Amostra 12 (Classe 3)
    [2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749], # Amostra 13 (Classe 4)
    [2.9034, 1.4640, 4.6061, 1.4598, 4.2912, 6.9142], # Amostra 14 (Classe 4)
    [3.0181, 1.4918, 4.7051, 1.3521, 4.2623, 6.7966], # Amostra 15 (Classe 4)
    [2.9374, 1.4896, 4.7219, 1.3977, 4.1863, 6.8336]  # Amostra 16 (Classe 4)
])

y_train = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])

# Test data (8 days)
X_test = np.array([
    [2.9817, 1.5656, 4.8391, 1.4311, 4.1916, 6.9718], # Dia 1
    [1.5537, 2.2615, 1.3169, 2.5873, 1.7570, 5.0958], # Dia 2
    [1.2240, 0.2445, 1.3595, 5.4192, 3.2027, 2.5675], # Dia 3
    [2.5828, 1.5146, 2.1119, 1.2859, 2.3414, 1.8695], # Dia 4
    [2.4168, 1.4857, 1.8959, 1.3013, 2.4500, 1.7868], # Dia 5
    [1.0604, 0.2276, 1.2806, 5.4732, 3.2133, 2.4839], # Dia 6
    [1.5246, 2.4254, 1.1353, 2.5325, 1.7569, 5.2640], # Dia 7
    [3.0565, 1.6259, 4.7743, 1.3654, 4.2904, 6.9808]  # Dia 8
])

# Run training
lvq = LVQ1(n_features=6, n_classes=4, alpha=0.05)
print("Pesos Iniciais (Vetores de Referência):")
lvq.initialize_weights(X_train, y_train)
for i, (w, c) in enumerate(zip(lvq.w, lvq.w_classes)):
    print(f"Classe {c}: {[round(val, 4) for val in w]}")

# We will train for 100 epochs
epochs = 100
lvq.train(X_train, y_train, epochs=epochs)

print("\nPesos Finais após o treinamento:")
for i, (w, c) in enumerate(zip(lvq.w, lvq.w_classes)):
    print(f"Classe {c}: {[round(val, 4) for val in w]}")

# Classify the test data
predictions = lvq.predict(X_test)
print("\nClassificação do Perfil de Potência dos Dias de Teste:")
for i, pred in enumerate(predictions):
    print(f"Dia {i+1}: Classe {pred}")
