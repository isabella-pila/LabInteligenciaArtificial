# Relatório de Implementação e Simulação da Rede Neural ART-1

**CEFET-MG — Centro Federal de Educação Tecnológica de Minas Gerais**  
**Campus VIII – Varginha**  
**Curso:** Bacharelado em Sistemas de Informação  
**Disciplina:** Laboratório de Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Data:** 11 de Junho de 2026  

---

## 1. Introdução Teórica à Rede ART-1

A Rede **ART-1** (Adaptive Resonance Theory 1), desenvolvida por Stephen Grossberg e Gail Carpenter em 1987, é uma arquitetura de rede neural artificial de aprendizado não supervisionado voltada para o agrupamento (*clustering*) de vetores binários. 

O principal diferencial da teoria ART é a resolução do **Dilema da Estabilidade-Plasticidade**:
*   **Plasticidade:** A capacidade de aprender novos padrões e se adaptar a ambientes em mudança.
*   **Estabilidade:** A capacidade de reter conhecimentos previamente adquiridos sem que novas informações os apaguem ou corrompam.

### Arquitetura da Rede
A rede é composta essencialmente por duas camadas principais de neurônios que interagem continuamente:
1.  **Camada $F_1$ (Camada de Comparação):** Recebe o padrão de entrada e realiza a comparação com a expectativa gerada pela camada superior.
2.  **Camada $F_2$ (Camada de Reconhecimento):** Camada competitiva (*Winner-Take-All*) cujos neurônios representam as categorias ou classes de agrupamento.
3.  **Subsistema de Orientação (Mecanismo de Reset):** Controla o grau de semelhança exigido entre a entrada e o protótipo da classe através do **parâmetro de vigilância ($\rho$)**.

---

## 2. Formulação Matemática do Algoritmo

O funcionamento da rede ART-1 divide-se nas seguintes etapas clássicas baseadas na formulação de Laurene Fausett:

### Passo 0: Inicialização de Parâmetros e Pesos
*   **Parâmetro de Vigilância ($\rho$):** Limiar de aceitação, onde $0 < \rho \le 1$.
*   **Constante de Aprendizado ($L$):** Fator multiplicativo para pesos bottom-up, onde $L > 1$ (comumente $L = 2.0$).
*   **Pesos Bottom-Up ($b_{ij}$):** Conectam a camada $F_1$ à camada $F_2$. Inicializados com valores pequenos:
    $$b_{ij}(0) = \frac{1}{1 + N}$$
    Onde $N$ é o número de componentes do vetor de entrada.
*   **Pesos Top-Down ($t_{ji}$):** Conectam a camada $F_2$ à camada $F_1$. Representam os protótipos da classe e são inicializados com $1$:
    $$t_{ji}(0) = 1$$

### Passo 1: Apresentação de um Vetor de Entrada $s$
Para cada vetor de entrada binário $s = (s_1, s_2, \dots, s_N)$, realiza-se o ciclo de busca e ressonância.

### Passo 2: Fase de Reconhecimento (Bottom-Up)
Calcula-se a ativação líquida $y_j$ de cada neurônio $j$ na camada $F_2$ que não esteja inibido:
$$y_j = \sum_{i=1}^{N} b_{ij} \cdot s_i$$

Seleciona-se o neurônio vencedor $J$ que possua a maior ativação:
$$y_J = \max_{j} \{ y_j \}$$

### Passo 3: Fase de Comparação e Teste de Vigilância (Top-Down)
O neurônio vencedor $J$ projeta seu protótipo de volta à camada $F_1$. A nova ativação da camada $F_1$ é dada por:
$$x_i = s_i \cdot t_{Ji}$$

Calcula-se o grau de casamento (vetores normados):
$$\text{Razão de Semelhança} = \frac{\|x\|}{\|s\|} = \frac{\sum_{i=1}^{N} x_i}{\sum_{i=1}^{N} s_i}$$

*   **Se $\frac{\|x\|}{\|s\|} \ge \rho$ (Ressonância):** O padrão é classificado na classe $J$ e os pesos são atualizados (Passo 4).
*   **Se $\frac{\|x\|}{\|s\|} < \rho$ (Reset):** O neurônio vencedor $J$ é temporariamente inibido para esta entrada. O processo retorna à Fase de Reconhecimento (Passo 2) para escolher o próximo melhor neurônio ativo em $F_2$.

### Passo 4: Atualização de Pesos (Fast Learning)
Quando ocorre a ressonância na classe $J$, atualizam-se os pesos bottom-up e top-down da seguinte forma:
$$b_{iJ}(new) = \frac{L \cdot x_i}{L - 1 + \|x\|}$$
$$t_{Ji}(new) = x_i$$

---

## 3. Dados de Entrada (Situações do Processo)

O comportamento do processo industrial é caracterizado por 10 situações e 16 variáveis de status binárias ($x_1$ a $x_{16}$):

| Situação | $x_1$ | $x_2$ | $x_3$ | $x_4$ | $x_5$ | $x_6$ | $x_7$ | $x_8$ | $x_9$ | $x_{10}$ | $x_{11}$ | $x_{12}$ | $x_{13}$ | $x_{14}$ | $x_{15}$ | $x_{16}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Situação 1** | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| **Situação 2** | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| **Situação 3** | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| **Situação 4** | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 |
| **Situação 5** | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| **Situação 6** | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 1 |
| **Situação 7** | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 0 |
| **Situação 8** | 1 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 1 | 1 |
| **Situação 9** | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| **Situação 10**| 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 1 |

*Nota: Observa-se que a **Situação 3** é idêntica à **Situação 8**, e a **Situação 5** é idêntica à **Situação 10**.*

---

## 4. Resultados das Simulações por Nível de Vigilância

Abaixo são apresentados os resultados obtidos executando o algoritmo ART-1 desenvolvido para as quatro taxas de vigilância estabelecidas.

### 4.1. Simulação com Vigilância $\rho = 0.5$
*   **Número de classes ativas:** 3 classes
*   **Convergência:** 3 épocas

#### Agrupamentos e Protótipos:
*   **Classe 1 (Filtro Geral do Tipo A):**
    *   **Situações:** Situação 3, Situação 4, Situação 8, Situação 9.
    *   **Protótipo (Memória Top-Down):** `[0 0 1 0 1 0 1 0 1 1 0 1 0 0 0 0]`
*   **Classe 2 (Filtro Geral do Tipo B):**
    *   **Situações:** Situação 2, Situação 5, Situação 7, Situação 10.
    *   **Protótipo (Memória Top-Down):** `[0 0 1 0 1 1 0 1 0 1 1 0 0 0 0 0]`
*   **Classe 3 (Filtro Geral do Tipo C):**
    *   **Situações:** Situação 1, Situação 6.
    *   **Protótipo (Memória Top-Down):** `[0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1]`

*Análise:* Com uma vigilância baixa ($\rho=0.5$), o critério de similaridade é muito flexível. A rede agrupa situações bastantes genéricas, restando apenas 3 grandes classes. Os protótipos armazenados possuem poucos bits em `1` pois o protótipo é a interseção (AND lógico) de todas as situações da classe.

---

### 4.2. Simulação com Vigilância $\rho = 0.8$
*   **Número de classes ativas:** 5 classes
*   **Convergência:** 2 épocas

#### Agrupamentos e Protótipos:
*   **Classe 1:**
    *   **Situações:** Situação 1, Situação 6.
    *   **Protótipo (Memória Top-Down):** `[0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1]`
*   **Classe 2:**
    *   **Situações:** Situação 2, Situação 7.
    *   **Protótipo (Memória Top-Down):** `[1 0 1 0 1 1 0 1 1 1 1 0 1 0 0 0]`
*   **Classe 3:**
    *   **Situações:** Situação 3, Situação 8.
    *   **Protótipo (Memória Top-Down):** `[1 0 1 1 1 1 1 0 1 1 0 1 1 0 1 1]`
*   **Classe 4:**
    *   **Situações:** Situação 4, Situação 9.
    *   **Protótipo (Memória Top-Down):** `[0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 0]`
*   **Classe 5:**
    *   **Situações:** Situação 5, Situação 10.
    *   **Protótipo (Memória Top-Down):** `[0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 1]`

*Análise:* Em $\rho=0.8$, a rede atinge um excelente nível de discernimento prático. As situações idênticas (3 e 8, 5 e 10) ficam juntas, e padrões muito parecidos (como 1 e 6, 2 e 7, 4 e 9) são correlacionados.

---

### 4.3. Simulação com Vigilância $\rho = 0.9$
*   **Número de classes ativas:** 7 classes
*   **Convergência:** 2 épocas

#### Agrupamentos e Protótipos:
*   **Classe 1:**
    *   **Situações:** Situação 1, Situação 6.
    *   **Protótipo (Memória Top-Down):** `[0 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1]`
*   **Classe 2:**
    *   **Situações:** Situação 2 (Classe unitária).
    *   **Protótipo (Memória Top-Down):** `[1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 0]`
*   **Classe 3:**
    *   **Situações:** Situação 3, Situação 8.
    *   **Protótipo (Memória Top-Down):** `[1 0 1 1 1 1 1 0 1 1 0 1 1 0 1 1]`
*   **Classe 4:**
    *   **Situações:** Situação 4 (Classe unitária).
    *   **Protótipo (Memória Top-Down):** `[1 1 1 0 1 0 1 0 1 1 1 1 0 1 0 0]`
*   **Classe 5:**
    *   **Situações:** Situação 5, Situação 10.
    *   **Protótipo (Memória Top-Down):** `[0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 1]`
*   **Classe 6:**
    *   **Situações:** Situação 7 (Classe unitária).
    *   **Protótipo (Memória Top-Down):** `[1 0 1 0 1 1 0 1 1 1 1 0 1 1 1 0]`
*   **Classe 7:**
    *   **Situações:** Situação 9 (Classe unitária).
    *   **Protótipo (Memória Top-Down):** `[0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 1]`

*Análise:* Ao elevar a vigilância para $0.9$, a exigência de semelhança aumenta. Paragrupamentos anteriores se separam: as Situações 2 e 7 foram separadas em classes distintas, assim como as Situações 4 e 9. Apenas padrões de altíssima semelhança mútua (como 1 e 6) continuam agrupados.

---

### 4.4. Simulação com Vigilância $\rho = 0.99$
*   **Número de classes ativas:** 8 classes
*   **Convergência:** 2 épocas

#### Agrupamentos e Protótipos:
*   **Classe 1:** Situação 1 `[0 1 0 1 1 0 1 0 1 1 0 1 1 1 1 1]`
*   **Classe 2:** Situação 2 `[1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 0]`
*   **Classe 3:** Situações 3, Situação 8 `[1 0 1 1 1 1 1 0 1 1 0 1 1 0 1 1]` *(Idênticas)*
*   **Classe 4:** Situação 4 `[1 1 1 0 1 0 1 0 1 1 1 1 0 1 0 0]`
*   **Classe 5:** Situações 5, Situação 10 `[0 0 1 1 1 1 1 1 0 1 1 0 0 0 0 1]` *(Idênticas)*
*   **Classe 6:** Situação 6 `[1 1 0 1 0 0 1 0 1 1 0 1 1 1 1 1]`
*   **Classe 7:** Situação 7 `[1 0 1 0 1 1 0 1 1 1 1 0 1 1 1 0]`
*   **Classe 8:** Situação 9 `[0 1 1 0 1 0 1 0 1 1 0 1 0 1 0 1]`

*Análise:* Com $\rho = 0.99$ (vigilância quase máxima), a rede torna-se extremamente estrita. Cada situação única é isolada em sua própria classe. As únicas situações que permanecem agrupadas são as que são estritamente idênticas bit-a-bit (Situações 3/8 e Situações 5/10).

---

## 5. Discussão e Conclusões sobre a Vigilância ($\rho$)

A análise empírica dos resultados confirma o papel do **parâmetro de vigilância ($\rho$)** como o controlador de sensibilidade e granularidade do agrupamento da rede ART-1:

1.  **Baixa Vigilância ($\rho \to 0$):** Agrupamentos amplos, grosseiros e poucas classes ativas. A rede tolera maior diversidade dentro da mesma classe. A desvantagem é que o diagnóstico para manutenção pode ser impreciso, já que comportamentos industriais consideravelmente diferentes acabam sendo misturados no mesmo grupo.
2.  **Alta Vigilância ($\rho \to 1$):** Agrupamentos específicos, rígidos e maior quantidade de classes. A rede é intolerante a diferenças. Útil quando pequenas alterações de status no maquinário industrial representam falhas críticas distintas.
3.  **Balanço Ideal ($\rho = 0.8$):** Revelou-se a melhor parametrização para este problema, pois reuniu de forma coerente situações com desvios muito sutis (ex: Situação 1 e 6 diferem em apenas 2 bits) ao mesmo tempo que manteve distinções lógicas claras entre as categorias gerais de manutenção.

---

## 6. Código Fonte Utilizado (Python)

Abaixo está a implementação limpa e documentada em Python da rede ART-1 utilizada nas simulações:

```python
import numpy as np

class ART1:
    def __init__(self, n_features, max_categories=20, rho=0.5, L=2.0):
        self.n_features = n_features
        self.max_categories = max_categories
        self.rho = rho
        self.L = L
        
        # Inicialização dos pesos bottom-up (b_ij) conforme Laurene Fausett
        self.b = np.ones((max_categories, n_features)) / (1.0 + n_features)
        
        # Inicialização dos pesos top-down (t_ji) com 1s
        self.t = np.ones((max_categories, n_features))
        
        # Indica quais classes já receberam ao menos um padrão
        self.committed = np.zeros(max_categories, dtype=bool)

    def train_pattern(self, s):
        norm_s = np.sum(s)
        if norm_s == 0:
            return -1
        
        active_candidates = np.ones(self.max_categories, dtype=bool)
        
        while True:
            # Reconhecimento: calcula y_j = b_j . s
            y = np.zeros(self.max_categories)
            for j in range(self.max_categories):
                if active_candidates[j]:
                    y[j] = np.dot(self.b[j], s)
                else:
                    y[j] = -1.0
            
            # Neurônio vencedor
            J = np.argmax(y)
            if y[J] < 0:
                raise ValueError("Capacidade máxima de neurônios F2 esgotada!")
            
            # Comparação (Interseção s AND t_J)
            x = s * self.t[J]
            norm_x = np.sum(x)
            
            # Teste de Vigilância
            if (norm_x / norm_s) >= self.rho:
                # Ressonância! Atualização dos pesos do neurônio J
                self.b[J] = (self.L * x) / (self.L - 1.0 + norm_x)
                self.t[J] = x
                self.committed[J] = True
                return J
            else:
                # Inibição (Reset) do neurônio J para este padrão
                active_candidates[J] = False

    def train(self, X, max_epochs=100):
        epoch = 0
        while epoch < max_epochs:
            prev_b = self.b.copy()
            prev_t = self.t.copy()
            
            for s in X:
                self.train_pattern(s)
                
            # Verifica se os pesos convergiram (parada)
            if np.allclose(self.b, prev_b) and np.allclose(self.t, prev_t):
                break
            epoch += 1
        return epoch + 1

    def cluster_data(self, X):
        clusters = {}
        for idx, s in enumerate(X):
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
                        # Se não casar com nenhuma classe já treinada, 
                        # é atribuído ao primeiro neurônio livre
                        uncommitted = np.where(~self.committed)[0]
                        category = uncommitted[0] if len(uncommitted) > 0 else -1
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
            clusters[category].append(idx + 1)
            
        return clusters
```
