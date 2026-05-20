import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def run_kmeans(X, K, max_iters=300):
    # Initialize with the first K training samples
    centers = X[:K].copy().astype(float)
    
    for iteration in range(max_iters):
        # Euclidean distance
        dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(dists, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        for j in range(K):
            cluster_points = X[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = np.mean(cluster_points, axis=0)
            else:
                # If a cluster is empty, keep previous center
                new_centers[j] = centers[j]
                
        if np.allclose(centers, new_centers, atol=1e-9):
            break
        centers = new_centers
        
    # Calculate variances: mean squared distance to center
    variances = []
    for j in range(K):
        cluster_points = X[labels == j]
        if len(cluster_points) > 0:
            var_j = np.mean(np.sum((cluster_points - centers[j])**2, axis=1))
        else:
            var_j = 1e-4 # Avoid division by zero
        variances.append(var_j)
        
    return centers, labels, np.array(variances)

def rbf_activation(x, center, variance, use_factor_two=True):
    dist_sq = np.sum((x - center)**2)
    if use_factor_two:
        return np.exp(-dist_sq / (2.0 * variance))
    else:
        return np.exp(-dist_sq / variance)

def train_output_layer(X_hid, d, eta=0.01, precision=1e-7, seed=42, max_epochs=1000000, bias_val=-1.0):
    N, num_features = X_hid.shape
    # Initialize weights randomly between 0 and 1
    np.random.seed(seed)
    w = np.random.rand(num_features)
    
    epoch = 0
    prev_eqm = float('inf')
    historico_eqm = []
    
    while epoch < max_epochs:
        epoch += 1
        eqm_atual = 0.0
        
        for k in range(N):
            x_k = X_hid[k]
            d_k = d[k]
            
            y_k = np.dot(w, x_k)
            error = d_k - y_k
            
            # Delta rule update
            w = w + eta * error * x_k
            
            eqm_atual += error**2
            
        eqm_atual = eqm_atual / N
        historico_eqm.append(eqm_atual)
        
        if abs(eqm_atual - prev_eqm) < precision:
            break
            
        prev_eqm = eqm_atual
        
    return w, epoch, eqm_atual, historico_eqm

def evaluate_network(X_test_hid, d_test, w):
    # Predict outputs
    y_pred = []
    relative_errors = []
    for k in range(len(d_test)):
        yk = np.dot(w, X_test_hid[k])
        y_pred.append(yk)
        
        # Relative error % = |d - y| / d * 100
        rel_err = (abs(d_test[k] - yk) / d_test[k]) * 100
        relative_errors.append(rel_err)
        
    mean_relative_error = np.mean(relative_errors)
    variance_relative_error = np.var(relative_errors)
    
    return np.array(y_pred), relative_errors, mean_relative_error, variance_relative_error

def run_experiment(use_factor_two=True, bias_val=-1.0):
    df_train, df_test = load_data('treinamento.csv', 'teste.csv')
    
    X_train = df_train[['x1', 'x2', 'x3']].values
    d_train = df_train['d'].values
    
    X_test = df_test[['x1', 'x2', 'x3']].values
    d_test = df_test['d'].values
    
    topologies = [5, 10, 15]
    seeds = [42, 43, 44] # T1, T2, T3
    
    # Store results
    # results[K][seed_idx] = {w, epoch, final_eqm, hist_eqm, y_pred, mean_rel, var_rel}
    results = {}
    
    for K in topologies:
        results[K] = []
        
        # 1. Run K-means to get centers and variances
        centers, labels, variances = run_kmeans(X_train, K)
        
        # Calculate hidden layer activations for train
        X_train_hid = []
        for x in X_train:
            phi = [rbf_activation(x, centers[j], variances[j], use_factor_two) for j in range(K)]
            X_train_hid.append([bias_val] + phi)
        X_train_hid = np.array(X_train_hid)
        
        # Calculate hidden layer activations for test
        X_test_hid = []
        for x in X_test:
            phi = [rbf_activation(x, centers[j], variances[j], use_factor_two) for j in range(K)]
            X_test_hid.append([bias_val] + phi)
        X_test_hid = np.array(X_test_hid)
        
        for t_idx, seed in enumerate(seeds):
            w, epochs, final_eqm, hist_eqm = train_output_layer(X_train_hid, d_train, eta=0.01, precision=1e-7, seed=seed, bias_val=bias_val)
            y_pred, relative_errors, mean_rel, var_rel = evaluate_network(X_test_hid, d_test, w)
            
            results[K].append({
                "w": w,
                "epochs": epochs,
                "final_eqm": final_eqm,
                "hist_eqm": hist_eqm,
                "y_pred": y_pred,
                "relative_errors": relative_errors,
                "mean_rel": mean_rel,
                "var_rel": var_rel
            })
            
    return results

def main():
    print("=== EXPERIMENTO RBF2: APROXIMAÇÃO DE FUNÇÃO (INJEÇÃO ELETRÔNICA) ===")
    
    # We will test the classical combination (Bias = -1, use_factor_two = True)
    # and print all the necessary tables.
    
    bias_val = -1.0
    use_factor_two = True
    
    print(f"Configuracao Principal: Bias = {bias_val}, RBF c/ fator 2 no denominador (classica)")
    results = run_experiment(use_factor_two=use_factor_two, bias_val=bias_val)
    
    # 1. Print training performance table (EQM and Epochs for T1, T2, T3 across K=5, K=10, K=15)
    print("\n--- TABELA 1: RESULTADOS DE TREINAMENTO (EQM e Épocas) ---")
    print("Treinamento\tRede 1 (K=5)\t\tRede 2 (K=10)\t\tRede 3 (K=15)")
    print("\t\tEQM\tÉpocas\tEQM\tÉpocas\tEQM\tÉpocas")
    for t_idx in range(3):
        r5 = results[5][t_idx]
        r10 = results[10][t_idx]
        r15 = results[15][t_idx]
        print(f"T{t_idx+1}\t\t{r5['final_eqm']:.6f}\t{r5['epochs']}\t{r10['final_eqm']:.6f}\t{r10['epochs']}\t{r15['final_eqm']:.6f}\t{r15['epochs']}")
        
    # 2. Print validation table
    df_test = pd.read_csv('teste.csv')
    X_test = df_test[['x1', 'x2', 'x3']].values
    d_test = df_test['d'].values
    
    print("\n--- TABELA 2: TABELA DE VALIDAÇÃO COM ERRO RELATIVO ---")
    header = "Amostra\tx1\tx2\tx3\td\t"
    header += "y1(T1)\ty1(T2)\ty1(T3)\t"
    header += "y2(T1)\ty2(T2)\ty2(T3)\t"
    header += "y3(T1)\ty3(T2)\ty3(T3)"
    print(header)
    
    for i in range(15):
        row_str = f"{i+1:02d}\t{X_test[i][0]:.4f}\t{X_test[i][1]:.4f}\t{X_test[i][2]:.4f}\t{d_test[i]:.4f}\t"
        preds = []
        for K in [5, 10, 15]:
            for t_idx in range(3):
                preds.append(f"{results[K][t_idx]['y_pred'][i]:.4f}")
        row_str += "\t".join(preds)
        print(row_str)
        
    # Print mean relative error and variance
    row_mean = "Erro Relativo Médio (%):\t\t\t\t"
    means = []
    for K in [5, 10, 15]:
        for t_idx in range(3):
            means.append(f"{results[K][t_idx]['mean_rel']:.3f}%")
    row_mean += "\t".join(means)
    print(row_mean)
    
    row_var = "Variância (%):\t\t\t\t"
    vars_ = []
    for K in [5, 10, 15]:
        for t_idx in range(3):
            vars_.append(f"{results[K][t_idx]['var_rel']:.3f}%")
    row_var += "\t".join(vars_)
    print(row_var)
    
    # 3. Find the best training run for each network (based on training EQM)
    # Plot learning curves on a single figure (3 plots, non-overlapping)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    topologies = [5, 10, 15]
    names = ["Rede 1 (N1=5)", "Rede 2 (N1=10)", "Rede 3 (N1=15)"]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']
    
    for idx, K in enumerate(topologies):
        # Best run based on training EQM
        eqms = [r['final_eqm'] for r in results[K]]
        best_t_idx = np.argmin(eqms)
        best_run = results[K][best_t_idx]
        
        axes[idx].plot(best_run['hist_eqm'], color=colors[idx], linewidth=2)
        axes[idx].set_title(f"{names[idx]} - Melhor: T{best_t_idx+1} ({best_run['epochs']} épocas)", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel("Época")
        axes[idx].set_ylabel("Erro Quadrático Médio (EQM)")
        axes[idx].set_yscale('log')
        axes[idx].grid(True, which="both", ls="--", alpha=0.5)
        
    plt.tight_layout()
    plt.savefig('rbf2_curvas_eqm.png', dpi=150)
    print("\nCurvas de EQM salvas em 'rbf2_curvas_eqm.png'")
    
    # Let's also do a run without the factor of 2 in the denominator to check if it converges better or worse
    print("\n--- EXECUÇÃO COM FORMULAÇÃO ALTERNATIVA (SEM FATOR 2 NO DENOMINADOR) ---")
    results_alt = run_experiment(use_factor_two=False, bias_val=-1.0)
    print("Treinamento\tRede 1 (K=5)\t\tRede 2 (K=10)\t\tRede 3 (K=15)")
    print("\t\tEQM\tÉpocas\tEQM\tÉpocas\tEQM\tÉpocas")
    for t_idx in range(3):
        r5 = results_alt[5][t_idx]
        r10 = results_alt[10][t_idx]
        r15 = results_alt[15][t_idx]
        print(f"T{t_idx+1}\t\t{r5['final_eqm']:.6f}\t{r5['epochs']}\t{r10['final_eqm']:.6f}\t{r10['epochs']}\t{r15['final_eqm']:.6f}\t{r15['epochs']}")
        
    # Relative errors for alternative
    print("Erro Relativo Médio (%):")
    means_alt = []
    for K in [5, 10, 15]:
        for t_idx in range(3):
            means_alt.append(f"K={K} T{t_idx+1}: {results_alt[K][t_idx]['mean_rel']:.3f}%")
    print("\n".join(means_alt))

if __name__ == "__main__":
    main()
