import numpy as np

# Conjunto de treinamento fornecido pelo usuário
# Formato: [x1, x2, x3, d]
dataset = np.array([
    [-0.6508, 0.1097, 4.0009, -1.0],
    [-1.4492, 0.8896, 4.4005, -1.0],
    [2.0850, 0.6876, 12.0710, -1.0],
    [0.2626, 1.1476, 7.7985, 1.0],
    [0.6418, 1.0234, 7.0427, 1.0],
    [0.2569, 0.6730, 8.3265, -1.0],
    [1.1155, 0.6043, 7.4446, 1.0],
    [0.0914, 0.3399, 7.0677, -1.0],
    [0.0121, 0.5256, 4.6316, 1.0],
    [-0.0429, 0.4660, 5.4323, 1.0],
    [0.4340, 0.6870, 8.2287, -1.0],
    [0.2735, 1.0287, 7.1934, 1.0],
    [0.4839, 0.4851, 7.4850, -1.0],
    [0.4089, -0.1267, 5.5019, -1.0],
    [1.4391, 0.1614, 8.5843, -1.0],
    [-0.9115, -0.1973, 2.1962, -1.0],
    [0.3654, 1.0475, 7.4858, 1.0],
    [0.2144, 0.7515, 7.1699, 1.0],
    [0.2013, 1.0014, 6.5489, 1.0],
    [0.6483, 0.2183, 5.8991, 1.0],
    [-0.1147, 0.2242, 7.2435, -1.0],
    [-0.7970, 0.8795, 3.8762, 1.0],
    [-1.0625, 0.6366, 2.4707, 1.0],
    [0.5307, 0.1285, 5.6883, 1.0],
    [-1.2200, 0.7777, 1.7252, 1.0],
    [0.3957, 0.1076, 5.6623, -1.0],
    [-0.1013, 0.5989, 7.1812, -1.0],
    [2.4482, 0.9455, 11.2095, 1.0],
    [2.0149, 0.6192, 10.9263, -1.0],
    [0.2012, 0.2611, 5.4631, 1.0]
])

# Separando entradas (x) e saídas desejadas (d)
# Adicionamos o valor de bias (x0 = 1) no início de cada vetor de entrada
# Portanto, a entrada será [1, x1, x2, x3]
num_patterns = dataset.shape[0]
X = np.c_[np.ones(num_patterns), dataset[:, :3]]
D = dataset[:, 3]

eta = 0.01  # Taxa de aprendizagem
max_epochs = 1000  # Limite máximo de épocas para evitar loop infinito caso não seja linearmente separável

def activation_function(v):
    # Função degrau bipolar: 1 se v >= 0, caso contrário -1
    return 1.0 if v >= 0.0 else -1.0

trained_weights = []

print(f"Iniciando treinamento do Perceptron (Regra de Hebb) com taxa de aprendizagem = {eta}\n")
print("-" * 60)

for i in range(1, 6):
    # Reiniciando o gerador de números aleatórios com sementes diferentes
    np.random.seed(i * 100)
    
    # Inicializando vetor de pesos com valores aleatórios entre 0 e 1
    # Pesos: [w0 (bias), w1, w2, w3]
    W = np.random.uniform(0, 1, 4)
    
    print(f"Treinamento {i}")
    print(f"Pesos iniciais: {W}")
    
    epochs = 0
    error_exists = True
    
    while error_exists and epochs < max_epochs:
        error_exists = False
        
        for k in range(num_patterns):
            x = X[k]
            d = D[k]
            
            # v = W * x
            v = np.dot(W, x)
            
            # y = f(v)
            y = activation_function(v)
            
            # Atualização dos pesos apenas quando houver erro
            if y != d:
                # Regra de Hebb supervisionada: W_novo = W_atual + eta * d * x
                W = W + eta * d * x
                error_exists = True
        
        epochs += 1
        
    print(f"Épocas necessárias: {epochs}")
    print(f"Pesos finais:     {W}")
    print("-" * 60)
    trained_weights.append(W)

# FASE DE TESTE
print("\nIniciando fase de teste com as novas amostras...\n")

test_data = np.array([
    [-0.3565, 0.0620, 5.9891],
    [-0.7842, 1.1267, 5.5912],
    [0.3012, 0.5611, 5.8234],
    [0.7757, 1.0648, 8.0677],
    [0.1570, 0.8028, 6.3040],
    [-0.7014, 1.0316, 3.6005],
    [0.3748, 0.1536, 6.1537],
    [-0.6920, 0.9404, 4.4058],
    [-1.3970, 0.7141, 4.9263],
    [-1.8842, -0.2805, 1.2548]
])

num_test_patterns = test_data.shape[0]
X_test = np.c_[np.ones(num_test_patterns), test_data]

# Cabeçalho da tabela
print(f"{'Amostra':^9} | {'x1':^9} | {'x2':^9} | {'x3':^9} | {'y (T1)':^8} | {'y (T2)':^8} | {'y (T3)':^8} | {'y (T4)':^8} | {'y (T5)':^8}")
print("-" * 105)

for k in range(num_test_patterns):
    x = X_test[k]
    
    # Calculando a saída para cada um dos 5 modelos treinados
    y_preds = []
    for w in trained_weights:
        v = np.dot(w, x)
        y = activation_function(v)
        y_preds.append(y)
        
    # Imprimindo a linha da tabela
    print(f"{k+1:^9} | {x[1]:^9.4f} | {x[2]:^9.4f} | {x[3]:^9.4f} | {y_preds[0]:^8.0f} | {y_preds[1]:^8.0f} | {y_preds[2]:^8.0f} | {y_preds[3]:^8.0f} | {y_preds[4]:^8.0f}")


