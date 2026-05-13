import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    # Clip to avoid overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights between 0 and 1
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.random.rand(self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.random.rand(self.output_size)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2

    def backward(self, X, y, output):
        m = X.shape[0]
        
        # Error in output
        delta2 = (y - output) * sigmoid_deriv(output)
        
        # Error in hidden layer
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_deriv(self.a1)
        
        # Gradients
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0) / m
        
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0) / m
        
        # Update weights (we are maximizing the fit, so we add the gradient for the Generalized Delta Rule)
        # Note: Delta rule formulation: W = W + eta * error * derivative * input
        self.W2 += self.learning_rate * dW2
        self.b2 += self.learning_rate * db2
        self.W1 += self.learning_rate * dW1
        self.b1 += self.learning_rate * db1

    def train(self, X, y, precision=1e-6, max_epochs=100000):
        # We need online or batch? The problem usually implies epoch (batch) or online. Let's do batch first or online?
        # "Regra Delta Generalizada" usually implies epoch-based or pattern-based. 
        # For small datasets, batch is common to calculate MSE. Let's do online training (stochastic) but calculate MSE over epoch.
        epochs = 0
        mse_history = []
        prev_mse = float('inf')
        
        while epochs < max_epochs:
            # Batch training
            # Forward
            output = self.forward(X)
            
            # Backward
            self.backward(X, y, output)
            
            # Compute MSE for epoch
            mse = np.mean(np.square(y - output))
            mse_history.append(mse)
            epochs += 1
            
            if abs(prev_mse - mse) <= precision:
                break
                
            prev_mse = mse
            
            if epochs % 1000 == 0:
                print(f"Epoch {epochs}, MSE: {mse:.6f}")
                
        return mse_history, epochs


def main():
    # Load data
    train_df = pd.read_csv('treinamento.csv')
    test_df = pd.read_csv('teste.csv')
    
    X_train = train_df[['x1', 'x2', 'x3']].values
    y_train = train_df[['d']].values
    
    X_test = test_df[['x1', 'x2', 'x3']].values
    y_test = test_df[['d']].values
    
    num_trainings = 5
    hidden_size = 5 # Standard guess since the figure is missing
    results = []
    
    best_models = []
    
    for t in range(num_trainings):
        print(f"Starting Training T{t+1}")
        # Fix seed for reproducibility
        np.random.seed(42 + t)
        
        mlp = MLP(input_size=3, hidden_size=hidden_size, output_size=1, learning_rate=0.1)
        mse_hist, epochs = mlp.train(X_train, y_train, precision=1e-6, max_epochs=50000)
        
        final_mse = mse_hist[-1]
        print(f"T{t+1} Finished: {epochs} epochs, MSE: {final_mse:.6f}")
        
        # Test validation
        y_pred = mlp.forward(X_test)
        
        results.append({
            'training': f"T{t+1}",
            'epochs': epochs,
            'mse': final_mse,
            'mse_hist': mse_hist,
            'y_pred': y_pred.flatten(),
            'mlp': mlp
        })

    # Part 2: Register results
    res_df = pd.DataFrame([{ 'Treinamento': r['training'], 'Erro Quadrático Médio': r['mse'], 'Número de Épocas': r['epochs'] } for r in results])
    res_df.to_csv('resultados_treinamento.csv', index=False)
    print("\nTabela de Treinamentos salva em resultados_treinamento.csv")
    print(res_df.to_string(index=False))
    
    # Part 3: Plot for the two trainings with largest number of epochs
    results.sort(key=lambda x: x['epochs'], reverse=True)
    t_a = results[0]
    t_b = results[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(t_a['mse_hist'])
    axes[0].set_title(f"EQM vs Épocas - {t_a['training']} ({t_a['epochs']} épocas)")
    axes[0].set_xlabel("Épocas")
    axes[0].set_ylabel("EQM")
    axes[0].grid(True)
    
    axes[1].plot(t_b['mse_hist'])
    axes[1].set_title(f"EQM vs Épocas - {t_b['training']} ({t_b['epochs']} épocas)")
    axes[1].set_xlabel("Épocas")
    axes[1].set_ylabel("EQM")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('graficos_eqm.png')
    print("\nGráficos salvos em graficos_eqm.png")
    
    # Part 5: Validation on test set
    # Sort back to T1, T2...
    results.sort(key=lambda x: x['training'])
    
    val_table = []
    
    for i in range(len(y_test)):
        row = {
            'Amostra': i+1,
            'x1': X_test[i][0],
            'x2': X_test[i][1],
            'x3': X_test[i][2],
            'd': y_test[i][0],
        }
        for j, r in enumerate(results):
            row[f"yrede(T{j+1})"] = r['y_pred'][i]
        val_table.append(row)
        
    val_df = pd.DataFrame(val_table)
    
    # Calculate Erro Relativo Médio (%) and Variância (%)
    # Erro relativo = |y_pred - d| / d * 100
    er_med = {}
    er_var = {}
    for j, r in enumerate(results):
        preds = r['y_pred']
        errors = np.abs(preds - y_test.flatten()) / y_test.flatten() * 100
        er_med[f"yrede(T{j+1})"] = np.mean(errors)
        er_var[f"yrede(T{j+1})"] = np.var(errors)
        
    er_row = {'Amostra': 'Erro Relativo Médio (%)', 'x1': '', 'x2': '', 'x3': '', 'd': ''}
    er_row.update(er_med)
    
    var_row = {'Amostra': 'Variância (%)', 'x1': '', 'x2': '', 'x3': '', 'd': ''}
    var_row.update(er_var)
    
    val_df = pd.concat([val_df, pd.DataFrame([er_row, var_row])], ignore_index=True)
    
    val_df.to_csv('validacao.csv', index=False)
    print("\nTabela de Validação salva em validacao.csv")
    print(val_df.to_string(index=False))

    # Identify best
    best_idx = np.argmin([er_med[f"yrede(T{j+1})"] for j in range(num_trainings)])
    print(f"\nMelhor configuração para generalização: T{best_idx+1}")

if __name__ == '__main__':
    main()
