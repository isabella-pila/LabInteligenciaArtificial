import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from kohonen_som import KohonenSOM

def plot_grid(coords, neuron_classes, wins, filename='mapa_topologico.png'):
    """
    Plots the 4x4 topological grid map of the SOM with harmonious colors.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Harmonious colors for Classes
    color_map = {
        'A': '#2ca02c',  # Green
        'B': '#ff7f0e',  # Orange
        'C': '#1f77b4',  # Blue
        'Vazio': '#f0f0f0'  # Light Grey
    }
    
    # Set limits and grid lines
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xticks(np.arange(0, 4))
    ax.set_yticks(np.arange(0, 4))
    ax.grid(True, which='both', color='#cccccc', linestyle='-', linewidth=2)
    
    # Invert y-axis to match typical matrix indices
    ax.invert_yaxis()
    
    for i in range(16):
        r, c = coords[i]
        cls = neuron_classes[i]
        color = color_map.get(cls, color_map['Vazio'])
        
        # Draw a beautiful patch/square for the neuron
        rect = plt.Rectangle((c - 0.5, r - 0.5), 1.0, 1.0, facecolor=color, edgecolor='#ffffff', linewidth=3, alpha=0.85)
        ax.add_patch(rect)
        
        # Write text info inside the cell
        votes_str = f"A:{int(wins[i,0])} B:{int(wins[i,1])} C:{int(wins[i,2])}"
        ax.text(c, r - 0.1, f"Neurônio {i+1}\nClasse: {cls}", ha='center', va='center', fontsize=12, fontweight='bold', color='#333333')
        ax.text(c, r + 0.25, votes_str, ha='center', va='center', fontsize=9, fontstyle='italic', color='#555555')
        
    ax.set_title("Mapa Auto-Organizável de Kohonen (Grid 4x4)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Topological grid plot saved as {filename}")

def main():
    # Load dataset paths
    train_path = 'treinamento.csv'
    test_path = 'teste.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: training or test CSV datasets are missing.")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df[['x1', 'x2', 'x3']].values
    X_test = test_df[['x1', 'x2', 'x3']].values
    
    # Ground truth classes for the 120 training samples
    # 1-20: Classe A (0)
    # 21-60: Classe B (1)
    # 61-120: Classe C (2)
    y_train = np.zeros(120, dtype=int)
    y_train[20:60] = 1  # B
    y_train[60:] = 2    # C
    
    # Instantiate and train SOM
    # Seed outside class for strict reproducibility
    np.random.seed(42)
    
    som = KohonenSOM(input_size=3, grid_rows=4, grid_cols=4, learning_rate=0.001, neighborhood_radius=1)
    print("Training Kohonen Self-Organizing Map...")
    som.train(X_train, epochs=5000, neighborhood_type='moore')
    
    # Evaluation: majority voting for each neuron
    wins = np.zeros((som.num_neurons, 3))
    for idx in range(len(X_train)):
        x_samp = X_train[idx]
        winner = som.get_winner(x_samp)
        wins[winner, y_train[idx]] += 1
        
    neuron_classes = {}
    for i in range(som.num_neurons):
        votes = wins[i]
        if np.sum(votes) == 0:
            neuron_classes[i] = 'Vazio'
        else:
            majority = np.argmax(votes)
            neuron_classes[i] = ['A', 'B', 'C'][majority]
            
    print("\nClass Mapping of Neurons in Grid:")
    for i in range(som.num_neurons):
        print(f"Neurônio {i+1} at {som.coords[i]}: Classe={neuron_classes[i]} (Ativações A:{int(wins[i,0])}, B:{int(wins[i,1])}, C:{int(wins[i,2])})")
        
    # Plot topological map
    plot_grid(som.coords, neuron_classes, wins, 'mapa_topologico.png')
    
    # Test set evaluation
    test_results = []
    active_neurons = np.where(np.sum(wins, axis=1) > 0)[0]
    
    for idx in range(len(X_test)):
        x_samp = X_test[idx]
        winner = som.get_winner(x_samp)
        pred_class = neuron_classes[winner]
        
        # Fallback in case a test pattern falls on an empty boundary neuron
        if pred_class == 'Vazio':
            dists = np.sum(np.square(som.W - x_samp), axis=1)
            closest_active = active_neurons[np.argmin(dists[active_neurons])]
            pred_class = neuron_classes[closest_active]
            winner = closest_active
            
        test_results.append({
            'Amostra': idx + 1,
            'x1': X_test[idx, 0],
            'x2': X_test[idx, 1],
            'x3': X_test[idx, 2],
            'Neurônio Vencedor': winner + 1,
            'Classe': pred_class
        })
        
    df_test_res = pd.DataFrame(test_results)
    df_test_res.to_csv('resultados_teste.csv', index=False)
    print("\nTest results saved successfully in resultados_teste.csv:")
    print(df_test_res.to_string(index=False))

if __name__ == '__main__':
    main()
