import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, momentum=0.9):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # Initialize weights randomly between 0 and 1
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.random.rand(self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.random.rand(self.output_size)
        
        # For momentum
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def set_weights(self, W1, b1, W2, b2):
        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()

    def get_weights(self):
        return self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        
        delta2 = (y - output) * sigmoid_deriv(output)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_deriv(self.a1)
        
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # Update with momentum
        self.v_W2 = self.momentum * self.v_W2 + self.learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + self.learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + self.learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + self.learning_rate * db1
        
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train(self, X, y, precision=1e-6, max_epochs=100000):
        epochs = 0
        mse_history = []
        prev_mse = float('inf')
        
        start_time = time.time()
        
        while epochs < max_epochs:
            output = self.forward(X)
            self.backward(X, y, output)
            
            mse = np.mean(np.square(y - output))
            mse_history.append(mse)
            epochs += 1
            
            if abs(prev_mse - mse) <= precision:
                break
                
            prev_mse = mse
            
        elapsed = time.time() - start_time
        return mse_history, epochs, elapsed

def plot_eqm_curves(mse_std, mse_mom, ep_std, ep_mom, filename='graficos_eqm.png'):
    """
    Decoupled plotting function to visualize Mean Squared Error curves side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(mse_std, color='#1f77b4', linewidth=2)
    axes[0].set_title(f"Padrão sem Momentum\n({ep_std} épocas)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Épocas", fontsize=10)
    axes[0].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    axes[1].plot(mse_mom, color='#d62728', linewidth=2)
    axes[1].set_title(f"Com Momentum = 0.9\n({ep_mom} épocas)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Épocas", fontsize=10)
    axes[1].set_ylabel("Erro Quadrático Médio (EQM)", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nGráficos de comparação salvos em {filename}")

def evaluate(model, X_test, y_test, name):
    """
    Decoupled, fully vectorized validation evaluation and table generation.
    """
    y_pred_raw = model.forward(X_test)
    y_pred_rounded = np.round(y_pred_raw).astype(int)
    
    # Vectorized accuracy: check rows where all elements match targets exactly
    matches = np.all(y_pred_rounded == y_test, axis=1)
    acc = np.mean(matches) * 100
    
    # Vectorized construction of DataFrame
    df = pd.DataFrame({
        'Amostra': np.arange(1, len(y_test) + 1),
        'x1': X_test[:, 0],
        'x2': X_test[:, 1],
        'x3': X_test[:, 2],
        'x4': X_test[:, 3],
        'd1': y_test[:, 0],
        'd2': y_test[:, 1],
        'd3': y_test[:, 2],
        'y1': y_pred_rounded[:, 0],
        'y2': y_pred_rounded[:, 1],
        'y3': y_pred_rounded[:, 2]
    })
    
    df.to_csv(f'validacao_{name}.csv', index=False)
    return acc

def main():
    # 1. Load Data
    train_df = pd.read_csv('treinamento.csv')
    test_df = pd.read_csv('teste.csv')
    
    X_train = train_df[['x1', 'x2', 'x3', 'x4']].values
    y_train = train_df[['d1', 'd2', 'd3']].values
    
    X_test = test_df[['x1', 'x2', 'x3', 'x4']].values
    y_test = test_df[['d1', 'd2', 'd3']].values
    
    # We need identical initial weights for both comparisons
    hidden_size = 5
    
    # Establish stochastic/random starting weights using a clean seed setup outside MLP class
    np.random.seed(42)
    dummy_mlp = MLP(4, hidden_size, 3)
    W1_init, b1_init, W2_init, b2_init = dummy_mlp.get_weights()
    
    # 2. Standard Backpropagation
    mlp_std = MLP(4, hidden_size, 3, learning_rate=0.1, momentum=0.0)
    mlp_std.set_weights(W1_init, b1_init, W2_init, b2_init)
    
    print("Iniciando treinamento Padrão...")
    mse_std, ep_std, time_std = mlp_std.train(X_train, y_train, precision=1e-6, max_epochs=100000)
    print(f"Padrão -> Épocas: {ep_std}, MSE Final: {mse_std[-1]:.6f}, Tempo: {time_std:.3f}s")
    
    # 3. Backpropagation with Momentum (0.9)
    mlp_mom = MLP(4, hidden_size, 3, learning_rate=0.1, momentum=0.9)
    mlp_mom.set_weights(W1_init, b1_init, W2_init, b2_init)
    
    print("Iniciando treinamento com Momentum...")
    mse_mom, ep_mom, time_mom = mlp_mom.train(X_train, y_train, precision=1e-6, max_epochs=100000)
    print(f"Momentum -> Épocas: {ep_mom}, MSE Final: {mse_mom[-1]:.6f}, Tempo: {time_mom:.3f}s")
    
    # 4. Save metrics to file
    with open("resultados_treinamento.txt", "w") as f:
        f.write("Treinamento Padrão:\n")
        f.write(f"Épocas: {ep_std}\n")
        f.write(f"EQM Final: {mse_std[-1]:.6f}\n")
        f.write(f"Tempo de Processamento: {time_std:.3f} segundos\n\n")
        
        f.write("Treinamento com Momentum (0.9):\n")
        f.write(f"Épocas: {ep_mom}\n")
        f.write(f"EQM Final: {mse_mom[-1]:.6f}\n")
        f.write(f"Tempo de Processamento: {time_mom:.3f} segundos\n")
        
    # Plotting (Decoupled call)
    plot_eqm_curves(mse_std, mse_mom, ep_std, ep_mom, 'graficos_eqm.png')
    
    # 5. Validation on test set (fully vectorized evaluation)
    acc_std = evaluate(mlp_std, X_test, y_test, "padrao")
    acc_mom = evaluate(mlp_mom, X_test, y_test, "momentum")
    
    with open("resultados_validacao.txt", "w") as f:
        f.write(f"Taxa de Acerto (Padrão): {acc_std:.2f}%\n")
        f.write(f"Taxa de Acerto (Momentum): {acc_mom:.2f}%\n")
        
    print(f"Validação: Padrão = {acc_std:.2f}%, Momentum = {acc_mom:.2f}%")

if __name__ == '__main__':
    main()
