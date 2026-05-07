import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_data(filepath):
    df = pd.read_csv(filepath)
    X = df[['x1', 'x2', 'x3', 'x4']].values
    d = df['d'].values
    # Adicionar o x0 = -1 para o bias (limiar)
    X_with_bias = np.insert(X, 0, -1, axis=1)
    return X_with_bias, d

def treinar_adaline(X, d, eta=0.0025, precisao=1e-6, max_epochs=100000):
    N, num_features = X.shape
    
    # Inicializar os pesos com valores aleatórios entre 0 e 1
    w = np.random.rand(num_features)
    pesos_iniciais = w.copy()
    
    epoca = 0
    eqm_anterior = float('inf')
    historico_eqm = []
    
    while epoca < max_epochs:
        epoca += 1
        eqm_atual = 0.0
        
        for k in range(N):
            x_k = X[k]
            d_k = d[k]
            
            # Saída linear u_k
            u_k = np.dot(w, x_k)
            
            # Atualização dos pesos (Regra Delta)
            w = w + eta * (d_k - u_k) * x_k
            
            # Acumular erro quadrado
            eqm_atual += (d_k - u_k)**2
            
        # Calcular Erro Quadrático Médio da época
        eqm_atual = eqm_atual / N
        historico_eqm.append(eqm_atual)
        
        # Verificar critério de parada
        if abs(eqm_atual - eqm_anterior) < precisao:
            break
            
        eqm_anterior = eqm_atual
        
    return pesos_iniciais, w, epoca, eqm_atual, historico_eqm

def main():
    filepath = os.path.join(os.path.dirname(__file__), 'treinamento.csv')
    X, d = load_data(filepath)
    
    print("=== TREINAMENTO DA REDE ADALINE ===")
    print(f"Taxa de aprendizado (eta): 0.0025")
    print(f"Precisão requerida (epsilon): 10^-6\n")
    
    historicos = []
    pesos_finais_list = []
    
    for i in range(5):
        # Reiniciar semente se desejar (não é estritamente necessário já que rand() continuará gerando valores novos,
        # mas garante números diferentes. Vamos deixar o gerador livre)
        pesos_iniciais, pesos_finais, epocas, eqm_final, historico_eqm = treinar_adaline(X, d)
        
        if i < 2:
            historicos.append(historico_eqm)
            
        pesos_finais_list.append(pesos_finais)
            
        print(f"--- Treinamento {i+1} ---")
        print(f"Pesos iniciais (w0...w4): {pesos_iniciais}")
        print(f"Pesos finais   (w0...w4): {pesos_finais}")
        print(f"Total de épocas: {epocas}")
        print(f"Erro Quadrático Médio final: {eqm_final:.6f}\n")
        
    # Plotar os gráficos dos dois primeiros treinamentos
    plt.figure(figsize=(10, 6))
    plt.plot(historicos[0], label='Treinamento 1')
    plt.plot(historicos[1], label='Treinamento 2')
    plt.title('Erro Quadrático Médio (EQM) vs Épocas de Treinamento')
    plt.xlabel('Época')
    plt.ylabel('EQM')
    plt.legend()
    plt.grid(True)
    
    # Salvar o gráfico na mesma pasta
    grafico_path = os.path.join(os.path.dirname(__file__), 'grafico_eqm.png')
    plt.savefig(grafico_path)
    print(f"Gráfico salvo com sucesso em: {grafico_path}\n")
    
    # --- TESTE DA REDE ADALINE ---
    teste_filepath = os.path.join(os.path.dirname(__file__), 'teste.csv')
    df_teste = pd.read_csv(teste_filepath)
    X_teste = df_teste[['x1', 'x2', 'x3', 'x4']].values
    # Adicionar bias
    X_teste_bias = np.insert(X_teste, 0, -1, axis=1)
    
    print("=== TESTE DA REDE ADALINE (Classificação das 15 amostras) ===")
    print("Amostra\ty(T1)\ty(T2)\ty(T3)\ty(T4)\ty(T5)")
    for i in range(X_teste_bias.shape[0]):
        x_k = X_teste_bias[i]
        saidas = []
        for w in pesos_finais_list:
            u = np.dot(w, x_k)
            y = 1 if u >= 0 else -1
            saidas.append(y)
        print(f"{i+1}\t{saidas[0]}\t{saidas[1]}\t{saidas[2]}\t{saidas[3]}\t{saidas[4]}")

if __name__ == "__main__":
    main()
