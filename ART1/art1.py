import numpy as np

class ART1:
    def __init__(self, n_features, max_categories=20, rho=0.5, L=2.0):
        self.n_features = n_features
        self.max_categories = max_categories
        self.rho = rho
        self.L = L
        
        # Initialize bottom-up weights b_ij: Fausett uses 1 / (1 + n_features)
        # where 0 < b_ij < L / (L - 1 + n_features)
        self.b = np.ones((max_categories, n_features)) / (1.0 + n_features)
        
        # Initialize top-down weights t_ji: all 1s
        self.t = np.ones((max_categories, n_features))
        
        # Track whether a category is committed (has been trained at least once)
        self.committed = np.zeros(max_categories, dtype=bool)

    def train_pattern(self, s):
        norm_s = np.sum(s)
        if norm_s == 0:
            # An all-zero vector matches nothing or goes to the first category
            return -1
        
        active_candidates = np.ones(self.max_categories, dtype=bool)
        
        while True:
            # Compute activations for uninhibited categories
            y = np.zeros(self.max_categories)
            for j in range(self.max_categories):
                if active_candidates[j]:
                    y[j] = np.dot(self.b[j], s)
                else:
                    y[j] = -1.0
            
            # Find the best candidate
            J = np.argmax(y)
            if y[J] < 0:
                raise ValueError("Out of category units!")
            
            # Vigilance test
            x = s * self.t[J]
            norm_x = np.sum(x)
            
            if (norm_x / norm_s) >= self.rho:
                # Resonance!
                # Update weights
                self.b[J] = (self.L * x) / (self.L - 1.0 + norm_x)
                self.t[J] = x
                self.committed[J] = True
                return J
            else:
                # Reset this candidate unit
                active_candidates[J] = False

    def train(self, X, max_epochs=100):
        epoch = 0
        while epoch < max_epochs:
            prev_b = self.b.copy()
            prev_t = self.t.copy()
            
            for s in X:
                self.train_pattern(s)
                
            # Convergence check: no change in weights
            if np.allclose(self.b, prev_b) and np.allclose(self.t, prev_t):
                break
            epoch += 1
        return epoch + 1

    def cluster_data(self, X):
        clusters = {}
        for idx, s in enumerate(X):
            # Find the winning category for the pattern
            # Note: during testing/final grouping, we use the trained weights
            norm_s = np.sum(s)
            if norm_s == 0:
                category = -1
            else:
                active_candidates = np.ones(self.max_categories, dtype=bool)
                category = -1
                while True:
                    y = np.zeros(self.max_categories)
                    for j in range(self.max_categories):
                        if active_candidates[j] and self.committed[j]:
                            y[j] = np.dot(self.b[j], s)
                        else:
                            y[j] = -1.0
                    
                    J = np.argmax(y)
                    if y[J] < 0:
                        # No committed category matched, so it would go to a new one
                        # Let's assign it to a new category if needed, or find the first uncommitted
                        uncommitted_indices = np.where(~self.committed)[0]
                        if len(uncommitted_indices) > 0:
                            category = uncommitted_indices[0]
                        else:
                            category = -1
                        break
                    
                    x = s * self.t[J]
                    norm_x = np.sum(x)
                    if (norm_x / norm_s) >= self.rho:
                        category = J
                        break
                    else:
                        active_candidates[J] = False
            
            if category not in clusters:
                clusters[category] = []
            clusters[category].append(idx + 1) # 1-based indexing for situations
            
        return clusters

# Situations data
situations = np.array([
    [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 1
    [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0], # Situação 2
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 3
    [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0], # Situação 4
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1], # Situação 5
    [1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1], # Situação 6
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0], # Situação 7
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], # Situação 8
    [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], # Situação 9
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1]  # Situação 10
])

rhos = [0.5, 0.8, 0.9, 0.99]

for rho in rhos:
    # We set max_categories = 20
    art = ART1(n_features=16, max_categories=20, rho=rho, L=2.0)
    epochs = art.train(situations, max_epochs=50)
    clusters = art.cluster_data(situations)
    
    # Filter and sort clusters
    active_clusters = {k: v for k, v in clusters.items() if k >= 0}
    num_classes = len(active_clusters)
    
    print(f"\n=========================================")
    print(f"Simulação com Grau de Vigilância (rho) = {rho}")
    print(f"Número de classes ativas: {num_classes}")
    print(f"Treinamento convergido em {epochs} épocas")
    print(f"Agrupamentos:")
    
    # Map cluster index to a simpler class name (e.g. Classe A, Classe B, etc.)
    sorted_keys = sorted(active_clusters.keys())
    for class_idx, key in enumerate(sorted_keys):
        sits = active_clusters[key]
        sit_str = ", ".join([f"Situação {s}" for s in sits])
        # Print prototype (top-down weight vector)
        proto_str = " ".join([str(int(val)) for val in art.t[key]])
        print(f"  - Classe {class_idx + 1}: {sit_str}")
        print(f"    Protótipo (Top-Down): [{proto_str}]")
