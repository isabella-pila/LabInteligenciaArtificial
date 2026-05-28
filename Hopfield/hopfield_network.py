import numpy as np

class HopfieldNetwork:
    """
    Classe para a Rede de Hopfield com N neurônios.
    Utiliza representação bipolar (-1 para pixel branco, +1 para pixel escuro).
    A matriz de pesos W é obtida pela regra do produto externo (Hebbiana) com diagonal nula.
    A função de ativação é equivalente à tangente hiperbólica com sigma muito grande (função sign).
    """
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.W = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        """
        Treina a rede usando a regra do produto externo (Hebbiana).
        W = \sum_{p=1}^P x^p (x^p)^T
        Com a diagonal forçada a zero para evitar auto-conexões e garantir estabilidade.
        """
        self.W = np.zeros((self.num_neurons, self.num_neurons))
        for p in patterns:
            p = np.array(p).flatten()
            if len(p) != self.num_neurons:
                raise ValueError(f"O tamanho do padrão ({len(p)}) deve ser igual ao número de neurônios ({self.num_neurons}).")
            self.W += np.outer(p, p)
        
        # Remove auto-conexões (diagonal nula)
        np.fill_diagonal(self.W, 0)
        
    def energy(self, state):
        """
        Calcula a energia do estado atual.
        E = -1/2 * \sum_i \sum_j w_{ij} * s_i * s_j = -1/2 * s^T * W * s
        """
        s = np.array(state).flatten()
        return -0.5 * np.dot(s, np.dot(self.W, s))

    def activation_function(self, v, current_state_val):
        """
        Função de ativação: Tangente Hiperbólica com sigma muito grande (tanh(sigma * v)).
        No limite de sigma -> infinito, a função tanh se comporta como a função sinal:
        - Para v > 0, tanh(sigma * v) = +1
        - Para v < 0, tanh(sigma * v) = -1
        - Para v == 0, mantemos o estado atual do neurônio (padrão em redes de Hopfield).
        """
        if v > 0:
            return 1.0
        elif v < 0:
            return -1.0
        else:
            return current_state_val

    def update_synchronous(self, state, max_iter=100):
        """
        Atualização Síncrona: todos os neurônios são atualizados ao mesmo tempo.
        s(t+1) = sign(W * s(t))
        Retorna o estado final estabilizado e o número de iterações.
        """
        s = np.array(state, dtype=float).copy()
        for iteration in range(max_iter):
            s_prev = s.copy()
            # Multiplicação paralela de todos os neurônios
            v = np.dot(self.W, s)
            
            # Aplicando a função de ativação
            for i in range(self.num_neurons):
                s[i] = self.activation_function(v[i], s_prev[i])
                
            # Verifica convergência
            if np.array_equal(s, s_prev):
                return s, iteration + 1
                
        return s, max_iter

    def update_asynchronous(self, state, max_iter=1000, random_order=True):
        """
        Atualização Assíncrona: um neurônio é atualizado de cada vez.
        Garante que a energia da rede decresça ou permaneça estável até atingir um mínimo local.
        - random_order: se True, embaralha a ordem de atualização dos neurônios a cada ciclo.
        Retorna o estado final estabilizado, o número de iterações (ciclos de varredura) e o histórico de energia.
        """
        s = np.array(state, dtype=float).copy()
        energy_history = [self.energy(s)]
        indices = np.arange(self.num_neurons)
        
        for iteration in range(max_iter):
            s_prev = s.copy()
            
            if random_order:
                np.random.shuffle(indices)
                
            for idx in indices:
                # Calcula v_i = \sum_j w_{ij} * s_j
                v_i = np.dot(self.W[idx], s)
                s[idx] = self.activation_function(v_i, s[idx])
            
            current_energy = self.energy(s)
            energy_history.append(current_energy)
            
            # Se nenhum neurônio mudou o seu estado após uma varredura completa, estabilizou
            if np.array_equal(s, s_prev):
                return s, iteration + 1, energy_history
                
        return s, max_iter, energy_history
