import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1.0 - x)

class TDNN:
    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.1, momentum=0.8):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Initialize weights randomly between 0 and 1 by default
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.random.rand(self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.random.rand(self.output_size)
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

    def init_weights(self, seed=None):
        """
        Re-initializes weights, optionally resetting the seed.
        """
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.rand(self.input_size, self.hidden_size)
        self.b1 = np.random.rand(self.hidden_size)
        self.W2 = np.random.rand(self.hidden_size, self.output_size)
        self.b2 = np.random.rand(self.output_size)
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)

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
        
        self.v_W2 = self.momentum * self.v_W2 + self.learning_rate * dW2
        self.v_b2 = self.momentum * self.v_b2 + self.learning_rate * db2
        self.v_W1 = self.momentum * self.v_W1 + self.learning_rate * dW1
        self.v_b1 = self.momentum * self.v_b1 + self.learning_rate * db1
        
        self.W2 += self.v_W2
        self.b2 += self.v_b2
        self.W1 += self.v_W1
        self.b1 += self.v_b1

    def train(self, X, y, precision=0.5e-6, max_epochs=100000):
        epochs = 0
        mse_history = []
        prev_mse = float('inf')
        
        while epochs < max_epochs:
            output = self.forward(X)
            self.backward(X, y, output)
            
            mse = np.mean(np.square(y - output))
            mse_history.append(mse)
            epochs += 1
            
            if abs(prev_mse - mse) <= precision:
                break
                
            prev_mse = mse
            
        return mse_history, epochs

def prepare_data(series, p):
    X = []
    y = []
    for i in range(len(series) - p):
        X.append(series[i:i+p])
        y.append(series[i+p])
    return np.array(X), np.array(y).reshape(-1, 1)

def plot_eqm_curves(best_runs, topologies, filename='graficos_eqm_topologias.png'):
    """
    Decoupled plotting function to visualize training Mean Squared Error for all topologies.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for idx, topo in enumerate(topologies):
        name = topo['name']
        best_run = best_runs[name]
        axes[idx].plot(best_run['mse_history'], color='#1f77b4', linewidth=2)
        axes[idx].set_title(f"{name} - {best_run['T']} (EQM Final: {best_run['eqm_final']:.6f})", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Épocas", fontsize=10)
        axes[idx].set_ylabel("EQM", fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nGráficos de EQM salvos em {filename}")

def plot_predictions(best_runs, topologies, test_series, t_axis, filename='graficos_estimativas_topologias.png'):
    """
    Decoupled plotting function to visualize Desired vs Predicted values for all topologies.
    """
    fig, axes = plt.subplots(3, 1, figsize=(8, 12))
    for idx, topo in enumerate(topologies):
        name = topo['name']
        best_run = best_runs[name]
        
        axes[idx].plot(t_axis, test_series, label="Desejado", marker='o', color='#2ca02c', linewidth=2)
        axes[idx].plot(t_axis, best_run['y_pred'], label="Estimado", marker='x', color='#d62728', linestyle='--', linewidth=2)
        axes[idx].set_title(f"{name} - {best_run['T']} Estimativas", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Tempo (t)", fontsize=10)
        axes[idx].set_ylabel("f(t)", fontsize=10)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, linestyle='--', alpha=0.7)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\nGráficos de estimativas salvos em {filename}")

def main():
    train_df = pd.read_csv('treinamento.csv')
    test_df = pd.read_csv('teste.csv')
    
    train_series = train_df['f'].values
    test_series = test_df['f'].values
    
    # 1. Topologies
    topologies = [
        {"name": "Rede 1", "p": 5, "N1": 10},
        {"name": "Rede 2", "p": 10, "N1": 15},
        {"name": "Rede 3", "p": 15, "N1": 25}
    ]
    
    results = {}
    
    # Run trainings
    # We will use fixed seeds outside the class for reproducibility across T1, T2, T3
    seeds = [42, 100, 2024]
    
    for topo in topologies:
        p = topo['p']
        N1 = topo['N1']
        name = topo['name']
        
        X_train, y_train = prepare_data(train_series, p)
        
        # Prepare test data: we need the last p elements from train to predict the first of test
        full_series = np.concatenate((train_series, test_series))
        # Test targets are at indices 100 to 119
        # So we create test inputs starting from index 100-p
        X_test = []
        y_test = []
        for i in range(100 - p, 120 - p):
            X_test.append(full_series[i:i+p])
            y_test.append(full_series[i+p])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test).reshape(-1, 1)
        
        results[name] = []
        
        print(f"Treinando {name} (p={p}, N1={N1})...")
        for t_idx, seed in enumerate(seeds):
            tdnn = TDNN(input_size=p, hidden_size=N1, output_size=1, learning_rate=0.1, momentum=0.8)
            tdnn.init_weights(seed)
            
            mse_history, epochs = tdnn.train(X_train, y_train, precision=0.5e-6, max_epochs=50000)
            
            # Evaluate on test
            y_pred = tdnn.forward(X_test)
            
            # Vectorized Erro relativo médio = mean(|y_des - y_pred| / y_des)
            errors = np.abs(y_test - y_pred) / y_test
            err_rel_med = np.mean(errors)
            variancia = np.var(errors)
            
            results[name].append({
                "T": f"T{t_idx+1}",
                "epochs": epochs,
                "eqm_final": mse_history[-1],
                "mse_history": mse_history,
                "y_pred": y_pred.flatten(),
                "err_rel_med": err_rel_med,
                "variancia": variancia
            })
            print(f"  T{t_idx+1} concluido. Épocas: {epochs}, EQM: {mse_history[-1]:.6f}")

    # Identify best runs per topology to plot
    best_runs = {}
    table_2_rows = []
    
    for topo in topologies:
        name = topo['name']
        res = results[name]
        
        # Sort to find best by lowest EQM final
        best_run = sorted(res, key=lambda x: x['eqm_final'])[0]
        best_runs[name] = best_run
        
        # Fill table 2
        table_2_rows.append({
            "Rede": name,
            "T1_EQM": res[0]['eqm_final'], "T1_Epocas": res[0]['epochs'],
            "T2_EQM": res[1]['eqm_final'], "T2_Epocas": res[1]['epochs'],
            "T3_EQM": res[2]['eqm_final'], "T3_Epocas": res[2]['epochs'],
        })

    # Plot EQM (Decoupled call)
    plot_eqm_curves(best_runs, topologies, 'graficos_eqm_topologias.png')
    
    # Plot Desired vs Estimated (Decoupled call)
    t_axis = np.arange(101, 121)
    plot_predictions(best_runs, topologies, test_series, t_axis, 'graficos_estimativas_topologias.png')
    
    # Generate and save tables
    df_table2 = pd.DataFrame(table_2_rows)
    df_table2.to_csv("tabela_treinamentos.csv", index=False)
    
    df_val = pd.DataFrame({"Amostra": [f"t = {101+i}" for i in range(20)], "f(t)": test_series})
    
    for name in ["Rede 1", "Rede 2", "Rede 3"]:
        for i in range(3):
            df_val[f"{name}_T{i+1}"] = results[name][i]['y_pred']
            
    # Fully vectorized structure for relative errors and variances rows
    er_med = ["Erro Relativo Médio:", ""]
    vari = ["Variância:", ""]
    for name in ["Rede 1", "Rede 2", "Rede 3"]:
        for i in range(3):
            er_med.append(results[name][i]['err_rel_med'])
            vari.append(results[name][i]['variancia'])
            
    df_val.loc[len(df_val)] = er_med
    df_val.loc[len(df_val)] = vari
    
    df_val.to_csv("tabela_validacao.csv", index=False)
    print("\nTabelas de treinamento e validação salvas com sucesso.")

if __name__ == '__main__':
    main()
