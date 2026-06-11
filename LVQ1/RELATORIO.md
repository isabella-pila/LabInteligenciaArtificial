# Relatório de Implementação e Simulação da Rede Neural LVQ-1

**CEFET-MG — Centro Federal de Educação Tecnológica de Minas Gerais**  
**Campus VIII – Varginha**  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 11 de Junho de 2026  

---

## 1. Introdução Teórica à Rede LVQ-1

A Rede **LVQ-1** (Learning Vector Quantization 1), proposta por Teuvo Kohonen, é um algoritmo de aprendizado híbrido (que combina aprendizado competitivo não supervisionado na camada interna com rotulação supervisionada) voltado para a classificação de padrões representados por vetores numéricos contínuos.

Diferente de redes MLP que criam hiperplanos separadores, a LVQ-1 define fronteiras de decisão por meio de **protótipos de classe** (vetores de referência ou *codebook vectors*) espalhados pelo espaço de entrada. A classificação de um novo padrão é feita com base no critério do **vizinho mais próximo** (menor distância euclidiana) em relação a esses protótipos.

### Arquitetura da Rede
A rede possui uma estrutura de três camadas simples:
1.  **Camada de Entrada:** Recebe as variáveis do padrão a ser classificado ($N$ neurônios).
2.  **Camada Competitiva (Kohonen):** Neurônios que competem entre si. Cada neurônio armazena um vetor de pesos (referência) de mesma dimensão da entrada.
3.  **Camada de Saída:** Associa linearmente cada neurônio da camada competitiva a uma das classes alvo.

---

## 2. Formulação Matemática do Algoritmo

O treinamento supervisionado da LVQ-1 ajusta a posição dos vetores de referência com base nas equações descritas por Laurene Fausett:

### Passo 0: Inicialização de Parâmetros e Pesos
*   **Taxa de Aprendizagem ($\alpha$):** Valor inicial pequeno, comumente definido como $\alpha = 0.05$ (conforme especificado na atividade).
*   **Inicialização dos Vetores de Referência ($w_j$):** 
    Utiliza-se o primeiro vetor de treinamento correspondente a cada classe como o vetor de referência inicial dessa classe:
    *   $w_1$ recebe a primeira amostra da **Classe 1** (Amostra 1)
    *   $w_2$ recebe a primeira amostra da **Classe 2** (Amostra 5)
    *   $w_3$ recebe a primeira amostra da **Classe 3** (Amostra 9)
    *   $w_4$ recebe a primeira amostra da **Classe 4** (Amostra 13)

### Passo 1: Apresentação de um Vetor de Entrada $x$ com Classe Alvo $T$
Para cada amostra de treinamento, calcula-se a distância euclidiana para todos os vetores de referência $w_j$:
$$d(x, w_j) = \sum_{i=1}^{N} (x_i - w_{ji})^2$$

### Passo 2: Seleção do Neurônio Vencedor $J$
Identifica-se o vetor de referência mais próximo (menor distância):
$$d(x, w_J) = \min_{j} \{ d(x, w_j) \}$$

Seja $C_J$ a classe associada ao vetor de referência vencedor $w_J$.

### Passo 3: Atualização dos Pesos
Os pesos do neurônio vencedor $J$ são ajustados dependendo do acerto ou erro de classificação:
*   **Se a classificação for correta ($C_J == T$):** O vetor de referência é atraído na direção do vetor de entrada para reforçar a representação da classe:
    $$w_J(new) = w_J(old) + \alpha \cdot (x - w_J(old))$$
*   **Se a classificação for incorreta ($C_J \ne T$):** O vetor de referência é repelido para longe do vetor de entrada para corrigir a fronteira:
    $$w_J(new) = w_J(old) - \alpha \cdot (x - w_J(old))$$

Os pesos dos outros neurônios permanecem inalterados.

### Passo 4: Redução da Taxa de Aprendizagem
A taxa de aprendizagem $\alpha$ é decrementada de forma linear ao final de cada época de treinamento para garantir a convergência fina dos pesos:
$$\alpha(epoch) = \alpha_0 \cdot \left(1 - \frac{epoch}{max\_epochs}\right)$$

---

## 3. Pesos Iniciais vs. Pesos Finais Treinados

Após o treinamento por **100 épocas** com taxa de aprendizagem inicial $\alpha = 0.05$ decaindo linearmente, os pesos convergiram da seguinte forma:

### Pesos Iniciais (Primeira Amostra de Cada Classe)
*   **Classe 1:** `[2.3976, 1.5328, 1.9044, 1.1937, 2.4184, 1.8649]`
*   **Classe 2:** `[1.1201, 0.0587, 1.3154, 5.3783, 3.1849, 2.4276]`
*   **Classe 3:** `[1.4871, 2.3448, 0.9918, 2.3160, 1.6783, 5.0850]`
*   **Classe 4:** `[2.9364, 1.5233, 4.6109, 1.3160, 4.2700, 6.8749]`

### Pesos Finais (Protótipos de Classe Treinados)
*   **Classe 1:** `[2.3416, 1.4867, 1.9422, 1.2460, 2.3309, 1.8146]`
*   **Classe 2:** `[1.0639, 0.1309, 1.2492, 5.3626, 3.1517, 2.3540]`
*   **Classe 3:** `[1.4053, 2.2805, 1.0353, 2.4219, 1.7345, 5.0962]`
*   **Classe 4:** `[2.9490, 1.4920, 4.6618, 1.3816, 4.2520, 6.8544]`

*Nota: É possível constatar que os vetores de referência finais se aproximaram muito da média aritmética (centroides) das amostras de cada classe, demonstrando que a rede conseguiu absorver com precisão o perfil médio de consumo/potência de cada uma das 4 classes de demanda.*

---

## 4. Classificação dos Perfis de Potência de Teste

Utilizando os pesos finais treinados, a rede classificou os 8 novos dias de teste com base na menor distância euclidiana em relação aos protótipos de classe:

| Dia | 7 horas | 8 horas | 9 horas | 10 horas | 11 horas | 12 horas | Classe Predita | Protótipo Vencedor Mais Próximo |
| :-: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 2.9817 | 1.5656 | 4.8391 | 1.4311 | 4.1916 | 6.9718 | **Classe 4** | `[2.9490, 1.4920, 4.6618, 1.3816, 4.2520, 6.8544]` |
| **2** | 1.5537 | 2.2615 | 1.3169 | 2.5873 | 1.7570 | 5.0958 | **Classe 3** | `[1.4053, 2.2805, 1.0353, 2.4219, 1.7345, 5.0962]` |
| **3** | 1.2240 | 0.2445 | 1.3595 | 5.4192 | 3.2027 | 2.5675 | **Classe 2** | `[1.0639, 0.1309, 1.2492, 5.3626, 3.1517, 2.3540]` |
| **4** | 2.5828 | 1.5146 | 2.1119 | 1.2859 | 2.3414 | 1.8695 | **Classe 1** | `[2.3416, 1.4867, 1.9422, 1.2460, 2.3309, 1.8146]` |
| **5** | 2.4168 | 1.4857 | 1.8959 | 1.3013 | 2.4500 | 1.7868 | **Classe 1** | `[2.3416, 1.4867, 1.9422, 1.2460, 2.3309, 1.8146]` |
| **6** | 1.0604 | 0.2276 | 1.2806 | 5.4732 | 3.2133 | 2.4839 | **Classe 2** | `[1.0639, 0.1309, 1.2492, 5.3626, 3.1517, 2.3540]` |
| **7** | 1.5246 | 2.4254 | 1.1353 | 2.5325 | 1.7569 | 5.2640 | **Classe 3** | `[1.4053, 2.2805, 1.0353, 2.4219, 1.7345, 5.0962]` |
| **8** | 3.0565 | 1.6259 | 4.7743 | 1.3654 | 4.2904 | 6.9808 | **Classe 4** | `[2.9490, 1.4920, 4.6618, 1.3816, 4.2520, 6.8544]` |

---

## 5. Conclusões e Análise Operacional

O modelo LVQ-1 demonstrou-se altamente eficaz na identificação e classificação de curvas de carga elétrica:
1.  **Consistência Temporal:** Perfis de potência apresentam comportamentos bem marcados em horários específicos. Por exemplo:
    *   **Classe 2:** Apresenta picos expressivos às 10 horas (potência $> 5.3$ kW) e declínio às 8 horas.
    *   **Classe 4:** É caracterizada por potências elevadas a partir das 9 horas, com ápice de consumo no horário do almoço (12 horas, $> 6.8$ kW).
2.  **Poder de Generalização:** Mesmo diante de flutuações e ruídos nas potências coletadas para os 8 dias de teste, a rede conseguiu associar cada dia de maneira unívoca a uma das classes de comportamento pré-estabelecidas no banco de treinamento.
3.  **Importância Prática:** Esse tipo de mapeamento permite que a concessionária de energia classifique o perfil de consumo diário de forma antecipada (avaliando apenas o início da manhã, como às 7h e 8h), servindo de subsídio para tomada de decisão em tempo real e despacho ótimo de geração de energia.

---

## 6. Código Fonte Utilizado (Python)

Abaixo está o código fonte implementado para as simulações e classificação da LVQ-1:

```python
import numpy as np

class LVQ1:
    def __init__(self, n_features, n_classes, alpha=0.05):
        self.n_features = n_features
        self.n_classes = n_classes
        self.alpha = alpha
        self.w = None
        self.w_classes = None

    def initialize_weights(self, X, y):
        # Inicialização: primeira amostra encontrada para cada classe no treino
        self.w = []
        self.w_classes = []
        for c in range(1, self.n_classes + 1):
            indices = np.where(y == c)[0]
            if len(indices) > 0:
                first_idx = indices[0]
                self.w.append(X[first_idx].copy())
                self.w_classes.append(c)
        self.w = np.array(self.w)
        self.w_classes = np.array(self.w_classes)

    def find_winner(self, x):
        # Distância Euclidiana
        distances = np.sum((self.w - x) ** 2, axis=1)
        winner_idx = np.argmin(distances)
        return winner_idx, distances[winner_idx]

    def train_pattern(self, x, target_class):
        winner_idx, _ = self.find_winner(x)
        winner_class = self.w_classes[winner_idx]
        
        # Regra de atualização do vetor de pesos
        if winner_class == target_class:
            self.w[winner_idx] += self.alpha * (x - self.w[winner_idx])
        else:
            self.w[winner_idx] -= self.alpha * (x - self.w[winner_idx])
        return winner_class

    def train(self, X, y, epochs=100):
        if self.w is None:
            self.initialize_weights(X, y)
            
        initial_alpha = self.alpha
        for epoch in range(epochs):
            for i in range(len(X)):
                self.train_pattern(X[i], y[i])
            # Decaimento linear do learning rate
            self.alpha = initial_alpha * (1.0 - (epoch / epochs))

    def predict(self, X):
        predictions = []
        for x in X:
            winner_idx, _ = self.find_winner(x)
            predictions.append(self.w_classes[winner_idx])
        return np.array(predictions)
```
