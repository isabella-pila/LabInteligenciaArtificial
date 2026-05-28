import numpy as np

class KohonenSOM:
    def __init__(self, input_size, grid_rows=4, grid_cols=4, learning_rate=0.001, neighborhood_radius=1):
        """
        Self-Organizing Map (SOM) or Kohonen Network.
        
        Parameters:
        -----------
        input_size : int
            Dimension of input patterns (number of features).
        grid_rows : int
            Number of rows in the 2D grid.
        grid_cols : int
            Number of cols in the 2D grid.
        learning_rate : float
            Learning rate (eta).
        neighborhood_radius : int
            Topological neighborhood radius.
        """
        self.input_size = input_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.learning_rate = learning_rate
        self.neighborhood_radius = neighborhood_radius
        self.num_neurons = grid_rows * grid_cols
        
        # Grid coordinates for topological distance calculations
        self.coords = np.array([[r, c] for r in range(grid_rows) for c in range(grid_cols)])
        
        # Weight initialization randomly in range [0, 1)
        self.W = np.random.rand(self.num_neurons, self.input_size)

    def get_winner(self, x):
        """
        Finds the Best Matching Unit (BMU) for input pattern x.
        Returns the index of the winning neuron.
        """
        # Fully vectorized Euclidean distance computation
        dists = np.sum(np.square(self.W - x), axis=1)
        return np.argmin(dists)

    def get_neighbors_moore(self, winner_idx):
        """
        Chebyshev/Moore distance neighborhood (horizontal, vertical, and diagonal).
        """
        winner_coord = self.coords[winner_idx]
        dists = np.max(np.abs(self.coords - winner_coord), axis=1)
        return np.where(dists <= self.neighborhood_radius)[0]

    def get_neighbors_manhattan(self, winner_idx):
        """
        Manhattan distance neighborhood (orthogonal only).
        """
        winner_coord = self.coords[winner_idx]
        dists = np.sum(np.abs(self.coords - winner_coord), axis=1)
        return np.where(dists <= self.neighborhood_radius)[0]

    def train_epoch(self, X, neighborhood_type='moore'):
        """
        Trains the SOM for one single epoch.
        """
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        for idx in indices:
            x_samp = X[idx]
            winner = self.get_winner(x_samp)
            
            # Select neighbor indices
            if neighborhood_type == 'moore':
                neighbors = self.get_neighbors_moore(winner)
            else:
                neighbors = self.get_neighbors_manhattan(winner)
                
            # Vectorized weight update step
            self.W[neighbors] += self.learning_rate * (x_samp - self.W[neighbors])

    def train(self, X, epochs=5000, neighborhood_type='moore'):
        """
        Runs the complete SOM training loop.
        """
        for epoch in range(epochs):
            self.train_epoch(X, neighborhood_type)
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{epochs} finished.")
