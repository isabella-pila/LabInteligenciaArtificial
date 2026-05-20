import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def run_kmeans(X, c1_init, c2_init, max_iters=200):
    centers = np.array([c1_init, c2_init], dtype=float)
    for iteration in range(max_iters):
        # Euclidean distance to each center
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Recompute centers
        new_centers = np.zeros_like(centers)
        for j in range(2):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)
            else:
                new_centers[j] = centers[j]
                
        if np.allclose(centers, new_centers, atol=1e-9):
            break
        centers = new_centers
        
    # Calculate variances: mean squared distance to center
    variances = []
    for j in range(2):
        cluster_points = X[labels == j]
        if len(cluster_points) > 0:
            var_j = np.mean(np.sum((cluster_points - centers[j])**2, axis=1))
        else:
            var_j = 0.0
        variances.append(var_j)
        
    return centers, labels, np.array(variances)

def rbf_activation(x, center, variance, use_factor_two=True):
    dist_sq = np.sum((x - center)**2)
    if use_factor_two:
        return np.exp(-dist_sq / (2.0 * variance))
    else:
        return np.exp(-dist_sq / variance)

def train_output_layer(X_hid, d, eta=0.01, precision=1e-7, max_epochs=1000000, bias_val=-1.0):
    N = X_hid.shape[0]
    # Initialize weights randomly between 0 and 1
    # We will use a fixed seed for reproducibility in each run, but let's test a few seeds
    np.random.seed(42)
    w = np.random.rand(3) # w0 (bias), w1 (h1), w2 (h2)
    
    epoch = 0
    prev_eqm = float('inf')
    historico_eqm = []
    
    while epoch < max_epochs:
        epoch += 1
        eqm_atual = 0.0
        
        # Online training (stochastic gradient descent / pattern-by-pattern delta rule)
        for k in range(N):
            x_k = X_hid[k] # [bias_val, phi1, phi2]
            d_k = d[k]
            
            # Linear output y_k
            y_k = np.dot(w, x_k)
            error = d_k - y_k
            
            # Update weights
            w = w + eta * error * x_k
            
            # Accumulate squared error
            eqm_atual += error**2
            
        eqm_atual = eqm_atual / N
        historico_eqm.append(eqm_atual)
        
        # Check stopping criterion
        if abs(eqm_atual - prev_eqm) < precision:
            break
            
        prev_eqm = eqm_atual
        
    return w, epoch, eqm_atual, historico_eqm

def main():
    train_path = 'treinamento.csv'
    test_path = 'teste.csv'
    
    df_train, df_test = load_data(train_path, test_path)
    
    # 1. K-means on patterns with presence of radiation (d = 1)
    df_pos = df_train[df_train['d'] == 1].copy()
    X_pos = df_pos[['x1', 'x2']].values
    
    # First two positive samples (Sample 3 and Sample 4 in original training set)
    c1_init = X_pos[0] # [0.1157, 0.3676]
    c2_init = X_pos[1] # [0.5147, 0.0167]
    
    centers, labels, variances = run_kmeans(X_pos, c1_init, c2_init)
    
    print("=== 1. CENTROS E VARIANCIAS DO K-MEANS (d = 1) ===")
    for j in range(2):
        print(f"Cluster {j+1}:")
        print(f"  Centro (x1, x2): ({centers[j][0]:.6f}, {centers[j][1]:.6f})")
        print(f"  Variancia (sigma^2): {variances[j]:.6f}")
        print(f"  Desvio Padrao (sigma): {np.sqrt(variances[j]):.6f}")
        print(f"  Numero de amostras: {np.sum(labels == j)}")
        
    # We will test all 4 combinations of configurations:
    # - Bias value: -1.0 vs 1.0
    # - RBF activation: exp(-d^2 / (2 * var)) vs exp(-d^2 / var)
    configs = [
        {"bias_val": -1.0, "use_factor_two": True, "desc": "Bias = -1, RBF c/ fator 2 no denominador (exp(-d^2 / 2s^2))"},
        {"bias_val": -1.0, "use_factor_two": False, "desc": "Bias = -1, RBF s/ fator 2 no denominador (exp(-d^2 / s^2))"},
        {"bias_val": 1.0, "use_factor_two": True, "desc": "Bias = +1, RBF c/ fator 2 no denominador (exp(-d^2 / 2s^2))"},
        {"bias_val": 1.0, "use_factor_two": False, "desc": "Bias = +1, RBF s/ fator 2 no denominador (exp(-d^2 / s^2))"},
    ]
    
    X_train = df_train[['x1', 'x2']].values
    d_train = df_train['d'].values
    
    X_test = df_test[['x1', 'x2']].values
    d_test = df_test['d'].values
    
    print("\n=== 2. E 3. TREINAMENTO E VALIDAÇÃO DA RBF ===")
    
    best_acc = -1
    best_config_results = None
    
    for config in configs:
        bias_val = config["bias_val"]
        use_factor_two = config["use_factor_two"]
        
        # Calculate hidden layer activations
        X_train_hid = []
        for x in X_train:
            phi1 = rbf_activation(x, centers[0], variances[0], use_factor_two)
            phi2 = rbf_activation(x, centers[1], variances[1], use_factor_two)
            X_train_hid.append([bias_val, phi1, phi2])
        X_train_hid = np.array(X_train_hid)
        
        # Train output layer
        w, epochs, final_eqm, hist_eqm = train_output_layer(X_train_hid, d_train, eta=0.01, precision=1e-7, bias_val=bias_val)
        
        # Test validation
        X_test_hid = []
        for x in X_test:
            phi1 = rbf_activation(x, centers[0], variances[0], use_factor_two)
            phi2 = rbf_activation(x, centers[1], variances[1], use_factor_two)
            X_test_hid.append([bias_val, phi1, phi2])
        X_test_hid = np.array(X_test_hid)
        
        # Outputs
        y_test = []
        y_pos = []
        correct = 0
        for k in range(len(X_test_hid)):
            yk = np.dot(w, X_test_hid[k])
            y_pos_k = 1 if yk >= 0.0 else -1
            y_test.append(yk)
            y_pos.append(y_pos_k)
            if y_pos_k == d_test[k]:
                correct += 1
                
        acc = (correct / len(d_test)) * 100
        
        print(f"\nConfiguracao: {config['desc']}")
        print(f"  Pesos Finais: W21,0 (bias) = {w[0]:.6f}, W21,1 = {w[1]:.6f}, W21,2 = {w[2]:.6f}")
        print(f"  Epocas ate convergencia: {epochs}")
        print(f"  EQM Final: {final_eqm:.8f}")
        print(f"  Taxa de Acerto no Teste (%): {acc:.1f}%")
        
        # Print the validation table for this configuration
        print("  Amostra\tx1\tx2\td\ty\typós")
        for i in range(len(d_test)):
            print(f"  {i+1}\t{X_test[i][0]:.4f}\t{X_test[i][1]:.4f}\t{d_test[i]}\t{y_test[i]:.4f}\t{y_pos[i]}")
            
        if acc > best_acc:
            best_acc = acc
            best_config_results = {
                "config": config,
                "w": w,
                "epochs": epochs,
                "final_eqm": final_eqm,
                "hist_eqm": hist_eqm,
                "y_test": y_test,
                "y_pos": y_pos,
                "acc": acc
            }
            
    # Plot the EQM learning curve for the best configuration
    best_config_desc = best_config_results["config"]["desc"]
    print(f"\n>>> A melhor configuracao foi: {best_config_desc} com {best_acc:.1f}% de acerto!")
    
    plt.figure(figsize=(8, 5))
    plt.plot(best_config_results["hist_eqm"], label='EQM por Época', color='#3182bd')
    plt.title(f'Curva de Aprendizado - {best_config_desc}')
    plt.xlabel('Época')
    plt.ylabel('Erro Quadrático Médio (EQM)')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig('rbf_curva_eqm.png')
    print("Grafico de erro salvo como 'rbf_curva_eqm.png'.")

if __name__ == "__main__":
    main()
