import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_train = pd.read_csv('treinamento.csv')
df_test = pd.read_csv('teste.csv')

# K-means centers and variances from our training run
c1 = np.array([0.164833, 0.612117])
c2 = np.array([0.398969, 0.157131])
var1 = 0.029806
var2 = 0.038460

# Weights for the best configuration (Bias = -1, use_factor_two = True)
w = np.array([1.002535, 2.377560, 2.697506])
bias_val = -1.0

# 1. Plot K-means clustering of positive samples
df_pos = df_train[df_train['d'] == 1].copy()
X_pos = df_pos[['x1', 'x2']].values

# Re-assign labels for plotting
dists = np.linalg.norm(X_pos[:, np.newaxis] - np.array([c1, c2]), axis=2)
labels = np.argmin(dists, axis=1)

plt.figure(figsize=(10, 8))
plt.scatter(X_pos[labels == 0, 0], X_pos[labels == 0, 1], c='#1f77b4', s=100, label='Cluster 1 (Radiação)', alpha=0.7)
plt.scatter(X_pos[labels == 1, 0], X_pos[labels == 1, 1], c='#ff7f0e', s=100, label='Cluster 2 (Radiação)', alpha=0.7)
plt.scatter([c1[0]], [c1[1]], c='red', marker='X', s=300, label='Centro 1', edgecolor='black')
plt.scatter([c2[0]], [c2[1]], c='darkred', marker='X', s=300, label='Centro 2', edgecolor='black')

# Draw circles for variances (1 standard deviation)
circle1 = plt.Circle((c1[0], c1[1]), np.sqrt(var1), color='#1f77b4', fill=False, linestyle='--', linewidth=2, label='1 Desvio Padrão C1')
circle2 = plt.Circle((c2[0], c2[1]), np.sqrt(var2), color='#ff7f0e', fill=False, linestyle='--', linewidth=2, label='1 Desvio Padrão C2')
plt.gca().add_patch(circle1)
plt.gca().add_patch(circle2)

plt.title('Treinamento da Camada Escondida (K-means - Apenas Radiação)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('rbf_clusters.png', dpi=150)
plt.close()

# 2. Plot Decision Boundary of the RBF Network
x1_grid, x2_grid = np.meshgrid(np.linspace(-0.1, 1.1, 200), np.linspace(-0.1, 1.1, 200))
grid_points = np.c_[x1_grid.ravel(), x2_grid.ravel()]

# Compute RBF activations for grid
phi1_grid = np.exp(-np.sum((grid_points - c1)**2, axis=1) / (2.0 * var1))
phi2_grid = np.exp(-np.sum((grid_points - c2)**2, axis=1) / (2.0 * var2))

# Network output y
y_grid = w[0] * bias_val + w[1] * phi1_grid + w[2] * phi2_grid
y_grid = y_grid.reshape(x1_grid.shape)

plt.figure(figsize=(10, 8))
# Fill decision regions
plt.contourf(x1_grid, x2_grid, y_grid, levels=[-999, 0, 999], colors=['#fddaec', '#decbe4'], alpha=0.3)
cs = plt.contour(x1_grid, x2_grid, y_grid, levels=[0], colors=['purple'], linewidths=2.5, linestyles='-')
plt.clabel(cs, inline=1, fontsize=12, fmt={0: 'Fronteira y=0'})

# Plot training points
train_pos = df_train[df_train['d'] == 1]
train_neg = df_train[df_train['d'] == -1]
plt.scatter(train_pos['x1'], train_pos['x2'], c='green', marker='o', s=80, label='Treino: Radiação (1)', edgecolor='black', alpha=0.7)
plt.scatter(train_neg['x1'], train_neg['x2'], c='red', marker='x', s=80, label='Treino: Ausência (-1)', alpha=0.7)

# Plot test points
test_pos = df_test[df_test['d'] == 1]
test_neg = df_test[df_test['d'] == -1]
plt.scatter(test_pos['x1'], test_pos['x2'], c='lightgreen', marker='o', s=150, linewidths=2, edgecolor='darkgreen', label='Teste: Radiação (1)', alpha=0.9)
plt.scatter(test_neg['x1'], test_neg['x2'], c='pink', marker='X', s=150, linewidths=2, edgecolor='red', label='Teste: Ausência (-1)', alpha=0.9)

# Highlight centers
plt.scatter([c1[0]], [c1[1]], c='yellow', marker='*', s=250, label='Centro RBF 1', edgecolor='black', zorder=5)
plt.scatter([c2[0]], [c2[1]], c='gold', marker='*', s=250, label='Centro RBF 2', edgecolor='black', zorder=5)

plt.title('Fronteira de Decisão da Rede RBF no Espaço de Entrada', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('rbf_decisao.png', dpi=150)
plt.close()

print("Plots successfully generated: rbf_clusters.png and rbf_decisao.png")
