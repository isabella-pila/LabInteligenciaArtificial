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
        
        # Will be initialized externally per training
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        
        self.v_W1 = None
        self.v_b1 = None
        self.v_W2 = None
        self.v_b2 = None

    def init_weights(self, seed):
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
    # We will use fixed seeds for reproducibility across T1, T2, T3
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
        # e.g. for predicting index 100 (which is t=101), we need full_series[100-p:100]
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
            # Erro relativo médio = mean(|y_des - y_pred| / y_des)
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

    # Generate EQM plot for best training per topology
    # Best training is the one with the lowest err_rel_med (or lowest EQM on training?). The prompt asks for "melhor treinamento realizado em cada uma delas". Usually it means lowest MSE on training or best generalization. I will select the one with lowest err_rel_med on test. Wait, the problem says "considerando o melhor treinamento {T1, T2 ou T3} realizado em cada uma". Let's pick the one with lowest EQM final.
    fig_eqm, axes_eqm = plt.subplots(3, 1, figsize=(8, 12))
    
    # Generate Desired vs Estimated plot for best training
    fig_est, axes_est = plt.subplots(3, 1, figsize=(8, 12))
    
    t_axis = np.arange(101, 121)
    
    table_2_rows = []
    val_table = {}
    
    for topo_idx, topo in enumerate(topologies):
        name = topo['name']
        res = results[name]
        
        # Sort to find best by lowest EQM final
        best_run = sorted(res, key=lambda x: x['eqm_final'])[0]
        
        # Fill table 2
        table_2_rows.append({
            "Rede": name,
            "T1_EQM": res[0]['eqm_final'], "T1_Epocas": res[0]['epochs'],
            "T2_EQM": res[1]['eqm_final'], "T2_Epocas": res[1]['epochs'],
            "T3_EQM": res[2]['eqm_final'], "T3_Epocas": res[2]['epochs'],
        })
        
        # Val table columns
        for i in range(3):
            val_table[f"{name}_T{i+1}"] = res[i]['y_pred']
            
        # Plot EQM
        axes_eqm[topo_idx].plot(best_run['mse_history'], color='blue')
        axes_eqm[topo_idx].set_title(f"{name} - {best_run['T']} (EQM Final: {best_run['eqm_final']:.6f})")
        axes_eqm[topo_idx].set_xlabel("Épocas")
        axes_eqm[topo_idx].set_ylabel("EQM")
        axes_eqm[topo_idx].grid(True)
        
        # Plot Desired vs Estimated
        axes_est[topo_idx].plot(t_axis, test_series, label="Desejado", marker='o')
        axes_est[topo_idx].plot(t_axis, best_run['y_pred'], label="Estimado", marker='x')
        axes_est[topo_idx].set_title(f"{name} - {best_run['T']} Estimativas")
        axes_est[topo_idx].set_xlabel("Tempo (t)")
        axes_est[topo_idx].set_ylabel("f(t)")
        axes_est[topo_idx].legend()
        axes_est[topo_idx].grid(True)
        
    fig_eqm.tight_layout()
    fig_eqm.savefig('graficos_eqm_topologias.png')
    
    fig_est.tight_layout()
    fig_est.savefig('graficos_estimativas_topologias.png')
    
    df_table2 = pd.DataFrame(table_2_rows)
    df_table2.to_csv("tabela_treinamentos.csv", index=False)
    
    df_val = pd.DataFrame({"Amostra": [f"t = {101+i}" for i in range(20)], "f(t)": test_series})
    
    for name in ["Rede 1", "Rede 2", "Rede 3"]:
        for i in range(3):
            df_val[f"{name}_T{i+1}"] = val_table[f"{name}_T{i+1}"]
            
    # Calculate Erro Relativo Médio and Variance
    er_med = ["Erro Relativo Médio:", ""]
    vari = ["Variância:", ""]
    for name in ["Rede 1", "Rede 2", "Rede 3"]:
        for i in range(3):
            er_med.append(results[name][i]['err_rel_med'])
            vari.append(results[name][i]['variancia'])
            
    df_val.loc[len(df_val)] = er_med
    df_val.loc[len(df_val)] = vari
    
    df_val.to_csv("tabela_validacao.csv", index=False)

if __name__ == '__main__':
    main()
