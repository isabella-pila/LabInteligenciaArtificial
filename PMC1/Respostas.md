# Respostas - Atividade PCM1 (Ressonância Magnética)

### 1. e 2. Execução de Treinamentos e Tabela de Resultados

A rede perceptron multicamadas foi treinada utilizando a Regra Delta Generalizada (backpropagation em batch), com inicialização de pesos aleatória (0 a 1), função de ativação logística (sigmoide), taxa de aprendizado $\eta = 0.1$, critério de parada por precisão $|EQM_{atual} - EQM_{anterior}| \leq 10^{-6}$ e até 50.000 épocas limite. Foram executados 5 treinamentos (T1 a T5), com sementes aleatórias distintas para garantir diferentes inicializações.

Abaixo estão os resultados obtidos de Erro Quadrático Médio (EQM) e Número de Épocas:

| Treinamento | Erro Quadrático Médio | Número de Épocas |
| :--- | :--- | :--- |
| 1º (T1) | 0.005203 | 17552 |
| 2º (T2) | 0.004398 | 12873 |
| 3º (T3) | 0.031058 | 333 |
| 4º (T4) | 0.004523 | 13457 |
| 5º (T5) | 0.004361 | 15234 |

---

### 3. Gráficos de Erro Quadrático Médio

Os dois treinamentos com maior número de épocas foram o **T1** (17.552 épocas) e o **T5** (15.234 épocas). O script gerou e salvou o gráfico de EQM versus Épocas não superpostos na mesma figura.

![Gráficos EQM vs Épocas](file:///c:/Users/isabe/Desktop/LabInteligenciaArtificial/PCM1/graficos_eqm.png)

---

### 4. Explicação sobre Variação do EQM e Épocas

**Baseado na tabela do item 2, explique de forma detalhada por que tanto o erro quadrático médio quanto o número de épocas variam de treinamento para treinamento:**

A variação ocorre porque o algoritmo *Backpropagation* baseia-se no método do Gradiente Descendente para atualizar os pesos, que é muito sensível ao **ponto de partida**. Como os pesos iniciais são gerados de forma **aleatória** (neste caso, entre 0 e 1) a cada treinamento, a rede neural começa sua busca pelo erro mínimo em locais diferentes da topologia (superfície) de erro. 

A superfície de erro de uma rede multicamadas é altamente não-linear e complexa, possuindo diversos mínimos locais, regiões de platô (onde o gradiente é quase nulo) e vales estreitos. Como resultado:
1. **Número de Épocas varia:** O caminho percorrido do ponto de partida até um ponto de convergência (onde $|EQM_t - EQM_{t-1}| \le 10^{-6}$) pode ser mais longo ou cruzar áreas de platô que atrasam o aprendizado (exigindo mais épocas, como em T1), ou pode parar muito rápido ao cair em um mínimo local raso logo de início (como ocorreu no T3 com apenas 333 épocas).
2. **Erro Quadrático Médio (EQM) varia:** Dependendo de qual bacia de atração o gradiente inicial cai, a rede converge para mínimos locais diferentes. Alguns mínimos locais são mais fundos (menor EQM final como o T5 e T2), enquanto outros resultam em um erro consideravelmente maior estagnado (como o T3).

---

### 5. Validação da Rede

Abaixo encontra-se a tabela validada com o conjunto de teste para as 5 redes, incluindo o erro relativo médio e variância dos mesmos:

| Amostra | x1 | x2 | x3 | d | yrede(T1) | yrede(T2) | yrede(T3) | yrede(T4) | yrede(T5) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0611 | 0.286 | 0.7464 | 0.4831 | 0.5295 | 0.5291 | 0.6249 | 0.5285 | 0.5186 |
| 2 | 0.5102 | 0.7464 | 0.086 | 0.5965 | 0.6123 | 0.6072 | 0.6331 | 0.5988 | 0.6118 |
| 3 | 0.0004 | 0.6916 | 0.5006 | 0.5318 | 0.5602 | 0.5644 | 0.6277 | 0.5522 | 0.5563 |
| 4 | 0.943 | 0.4476 | 0.2648 | 0.6843 | 0.6852 | 0.6795 | 0.6358 | 0.6842 | 0.6878 |
| 5 | 0.1399 | 0.161 | 0.2477 | 0.2872 | 0.3898 | 0.3770 | 0.6219 | 0.3799 | 0.3740 |
| 6 | 0.6423 | 0.3229 | 0.8567 | 0.7663 | 0.7059 | 0.7120 | 0.6327 | 0.7182 | 0.7114 |
| 7 | 0.6492 | 0.0007 | 0.6422 | 0.5666 | 0.5892 | 0.5800 | 0.6291 | 0.5944 | 0.5814 |
| 8 | 0.1818 | 0.5078 | 0.9046 | 0.6601 | 0.6547 | 0.6648 | 0.6295 | 0.6631 | 0.6566 |
| 9 | 0.7382 | 0.2647 | 0.1916 | 0.5427 | 0.5749 | 0.5588 | 0.6314 | 0.5643 | 0.5660 |
| 10 | 0.3879 | 0.1307 | 0.8656 | 0.5836 | 0.6062 | 0.6067 | 0.6279 | 0.6156 | 0.6015 |
| 11 | 0.1903 | 0.6523 | 0.782 | 0.695 | 0.6638 | 0.6739 | 0.6307 | 0.6691 | 0.6672 |
| 12 | 0.8401 | 0.449 | 0.2719 | 0.679 | 0.6641 | 0.6576 | 0.6347 | 0.6614 | 0.6649 |
| 13 | 0.0029 | 0.3264 | 0.2476 | 0.2956 | 0.3958 | 0.3868 | 0.6219 | 0.3856 | 0.3813 |
| 14 | 0.7088 | 0.9342 | 0.2763 | 0.7742 | 0.7328 | 0.7378 | 0.6375 | 0.7328 | 0.7414 |
| 15 | 0.1283 | 0.1882 | 0.7253 | 0.4662 | 0.5163 | 0.5126 | 0.6246 | 0.5152 | 0.5035 |
| 16 | 0.8882 | 0.3077 | 0.8931 | 0.8093 | 0.7522 | 0.7594 | 0.6352 | 0.7650 | 0.7611 |
| 17 | 0.2225 | 0.9182 | 0.782 | 0.7581 | 0.7230 | 0.7366 | 0.6337 | 0.7303 | 0.7311 |
| 18 | 0.1957 | 0.8423 | 0.3085 | 0.5826 | 0.6056 | 0.6088 | 0.6310 | 0.5958 | 0.6067 |
| 19 | 0.9991 | 0.5914 | 0.3933 | 0.7938 | 0.7444 | 0.7460 | 0.6378 | 0.7475 | 0.7518 |
| 20 | 0.2299 | 0.1524 | 0.7353 | 0.5012 | 0.5378 | 0.5336 | 0.6256 | 0.5387 | 0.5264 |
| **Erro Relativo Médio (%)** | - | - | - | - | **8.087%** | **7.210%** | **25.135%** | **7.219%** | **6.711%** |
| **Variância (%)** | - | - | - | - | **86.079** | **69.330** | **928.914** | **71.882** | **62.206** |

---

### 6. Indicação da Melhor Configuração

**Baseado nas análises da tabela acima indique qual das configurações finais de treinamento {T1, T2, T3, T4 ou T5} seria a mais adequada para o sistema de ressonância magnética, ou seja, qual delas está oferecendo a melhor generalização:**

A configuração mais adequada é o **T5**. Ao analisarmos a tabela de validação, observa-se que T5 obteve o menor Erro Relativo Médio (6,71%) em todo o conjunto de teste de dados inéditos (além da menor variância, 62,20%). 

Isso significa que, além de cometer o menor erro percentual em média para as novas amostras, o modelo manteve as predições mais consistentes e com menor oscilação de erro entre os cenários simulados. Embora os modelos T2 e T5 tenham tido EQMs de treinamento muito parecidos (~0.0043), o **T5** superou os demais frente aos dados não vistos, atestando sua superior capacidade de **generalização** que é de fundamental importância para o sistema de processamento de imagens por ressonância magnética real.
