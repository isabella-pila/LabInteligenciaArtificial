import numpy as np
from hopfield_network import HopfieldNetwork

# Símbolos para representação premium no terminal
BLACK_PIXEL = "█"  # Para +1 (pixel escuro)
WHITE_PIXEL = "░"  # Para -1 (pixel branco)

# Definição dos 4 padrões originais de 9x5 (45 bits)
PATTERNS_GRID = {
    1: [
        "..##.",
        ".###.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##.",
        "..##."
    ],
    2: [
        "#####",
        "#####",
        "...##",
        "...##",
        "#####",
        "##...",
        "##...",
        "#####",
        "#####"
    ],
    3: [
        "#####",
        "#####",
        "...##",
        "...##",
        "#####",
        "...##",
        "...##",
        "#####",
        "#####"
    ],
    4: [
        "##.##",
        "##.##",
        "##.##",
        "#####",
        "#####",
        "...##",
        "...##",
        "...##",
        "...##"
    ]
}

def grid_to_vector(grid):
    """Converte uma grade de caracteres ('.' e '#') em um vetor bipolar de floats (-1.0 e 1.0)"""
    vec = []
    for row in grid:
        for char in row:
            if char in ('#', '1', 'x', '█'):
                vec.append(1.0)
            else:
                vec.append(-1.0)
    return np.array(vec, dtype=float)

def vector_to_grid_lines(vec, rows=9, cols=5):
    """Converte um vetor bipolar em linhas de caracteres formatadas para exibição no terminal"""
    lines = []
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            idx = r * cols + c
            val = vec[idx]
            row_str += BLACK_PIXEL if val > 0 else WHITE_PIXEL
        lines.append(row_str)
    return lines

def add_exact_noise(vector, noise_pct=0.20):
    """Fliciona exatamente noise_pct% de pixels aleatoriamente no vetor bipolar"""
    n_neurons = len(vector)
    n_flip = int(round(n_neurons * noise_pct))
    
    noisy_vector = vector.copy()
    flip_indices = np.random.choice(n_neurons, size=n_flip, replace=False)
    noisy_vector[flip_indices] *= -1.0
    return noisy_vector, flip_indices

def print_side_by_side(orig_vec, noisy_vec, rec_vec, label="Simulação"):
    """Exibe os três padrões lado a lado com formatação elegante"""
    orig_lines = vector_to_grid_lines(orig_vec)
    noisy_lines = vector_to_grid_lines(noisy_vec)
    rec_lines = vector_to_grid_lines(rec_vec)
    
    print(f"\n{label}")
    print(f"{'Transmitida':^9}      {'Distorcida':^9}      {'Recuperada':^9}")
    print(f"{'(Sem Ruído)':^9}      {'(20% Ruído)':^9}      {'(Hopfield)':^9}")
    print("-" * 45)
    for i in range(9):
        print(f"  {orig_lines[i]}          {noisy_lines[i]}          {rec_lines[i]}")
    print("-" * 45)

def run_12_simulations():
    """Simula 12 situações de transmissão (3 para cada um dos 4 padrões) com 20% de ruído"""
    print("=" * 60)
    print("  SIMULAÇÃO DE 12 SITUAÇÕES DE TRANSMISSÃO COM 20% DE RUÍDO")
    print("=" * 60)
    
    # Inicializa e treina a rede
    num_neurons = 45
    net = HopfieldNetwork(num_neurons)
    
    # Carrega os vetores dos padrões
    patterns = {p_id: grid_to_vector(grid) for p_id, grid in PATTERNS_GRID.items()}
    net.train(list(patterns.values()))
    
    # Define sementes fixas para reprodutibilidade das 3 simulações de cada padrão
    seeds = {
        1: [42, 101, 2023],
        2: [123, 456, 789],
        3: [999, 888, 777],
        4: [111, 222, 333]
    }
    
    results_summary = []
    
    for p_id in [1, 2, 3, 4]:
        orig_vec = patterns[p_id]
        print(f"\n>>> PADRÃO {p_id} (Dígito {p_id})")
        
        for sim_idx in range(3):
            # Define a semente para garantir resultados consistentes e reprodutíveis
            np.random.seed(seeds[p_id][sim_idx])
            
            # Adiciona exatamente 20% de ruído (9 pixels invertidos)
            noisy_vec, flipped = add_exact_noise(orig_vec, 0.20)
            
            # Recupera usando atualização assíncrona
            recovered_vec, steps, energy_hist = net.update_asynchronous(noisy_vec)
            
            # Verifica se a recuperação foi perfeita
            success = np.array_equal(recovered_vec, orig_vec)
            
            # Imprime os resultados no terminal
            label = f"Caso {sim_idx+1} (Padrão {p_id}) - Iterações: {steps} | Energia Final: {energy_hist[-1]:.1f}"
            print_side_by_side(orig_vec, noisy_vec, recovered_vec, label)
            print(f"Pixels corrompidos (índices): {sorted(flipped.tolist())}")
            print(f"Sucesso na recuperação? {'SIM (100% fiel)' if success else 'NÃO (Espúrio ou Reverso)'}")
            
            results_summary.append({
                "padrao": p_id,
                "caso": sim_idx + 1,
                "iteracoes": steps,
                "energia_inicial": energy_hist[0],
                "energia_final": energy_hist[-1],
                "sucesso": success
            })
            
    return results_summary

def run_noise_sensitivity_experiment(n_trials=100):
    """
    Experimento para o Item 3: analisa o comportamento da rede com níveis crescentes de ruído.
    Mede a taxa de recuperação perfeita dos padrões originais para ruídos de 0% a 100%.
    """
    print("\n" + "=" * 60)
    print("   EXPERIMENTO DE SENSIBILIDADE AO RUÍDO (0% a 100%)")
    print("=" * 60)
    
    num_neurons = 45
    net = HopfieldNetwork(num_neurons)
    patterns = {p_id: grid_to_vector(grid) for p_id, grid in PATTERNS_GRID.items()}
    net.train(list(patterns.values()))
    
    noise_levels = np.arange(0, 1.05, 0.10)
    print(f"{'Nível de Ruído (%)':^20} | {'Taxa de Sucesso (%)':^22} | {'Estado Comum de Falha':^25}")
    print("-" * 75)
    
    # Fixar semente para o experimento quantitativo
    np.random.seed(9876)
    
    for nl in noise_levels:
        successes = 0
        total_runs = 0
        failure_types = {"reverso": 0, "espurio": 0}
        
        for p_id, orig_vec in patterns.items():
            for _ in range(n_trials // 4): # Distribui igualmente entre os 4 padrões
                noisy_vec, _ = add_exact_noise(orig_vec, nl)
                recovered_vec, _, _ = net.update_asynchronous(noisy_vec)
                
                if np.array_equal(recovered_vec, orig_vec):
                    successes += 1
                else:
                    # Classifica a falha
                    if np.array_equal(recovered_vec, -orig_vec):
                        failure_types["reverso"] += 1
                    else:
                        failure_types["espurio"] += 1
                total_runs += 1
                
        success_rate = (successes / total_runs) * 100
        
        # Determina o estado de falha predominante
        if success_rate == 100.0:
            failure_str = "Nenhum (100% de sucesso)"
        elif failure_types["reverso"] > failure_types["espurio"]:
            failure_str = "Estado Reverso (-Padrão)"
        else:
            failure_str = "Estado Espúrio (Mín. Local)"
            
        print(f"{nl*100:^18.0f}% | {success_rate:^20.1f}% | {failure_str:^25}")

if __name__ == "__main__":
    # Corrige formatação de strings de padrões se houver aspas duplicadas ou quebradas por engano
    # Ajustando o Pattern 2 e 3 que podem ter problemas de digitação
    PATTERNS_GRID[2][3] = "...##"
    PATTERNS_GRID[3][6] = "...##"
    
    # Roda as simulações e o experimento
    run_12_simulations()
    run_noise_sensitivity_experiment(200)
