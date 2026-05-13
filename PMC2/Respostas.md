# Respostas - Atividade PCM2 (Classificador de Conservantes)

### 1 e 2. Treinamentos e Gráficos (Padrão vs Momentum)

Foi construída uma rede neural do tipo Perceptron Multicamadas (MLP) com 4 entradas ($x_1, x_2, x_3, x_4$) e 3 saídas ($y_1, y_2, y_3$) para realizar a classificação do conservante. A topologia de teste definiu 5 neurônios na camada oculta. Foram executados dois treinamentos sob as mesmas condições e matrizes de pesos iniciais, com precisão limite estipulada em $|EQM_{atual} - EQM_{anterior}| \leq 10^{-6}$:

1. **Treinamento Padrão** (Regra Delta Generalizada simples, $\eta = 0.1$):
   - **Épocas até a convergência:** 25.203
   - **Erro Quadrático Médio Final:** 0.028284
   - **Tempo de Processamento:** ~6.01 segundos

2. **Treinamento com Momentum** ($\eta = 0.1$, Fator de Momentum $\alpha = 0.9$):
   - **Épocas até a convergência:** 5.228
   - **Erro Quadrático Médio Final:** 0.019362
   - **Tempo de Processamento:** ~0.98 segundos

O uso do momentum apresentou uma redução drástica tanto no número de épocas quanto no tempo de processamento em comparação ao método padrão, garantindo além de tudo um Erro Quadrático Médio menor ao final.

Abaixo encontram-se os respectivos gráficos impressos lado-a-lado:

![Gráficos EQM Padrão e Momentum](file:///c:/Users/isabe/Desktop/LabInteligenciaArtificial/PCM2/graficos_eqm.png)

---

### 3. Rotina de Pós-Processamento

Uma vez que o problema é de classificação com codificação $one-hot$, as saídas contínuas geradas pela ativação sigmoide (reais entre 0 e 1) devem ser convertidas para os inteiros `0` ou `1`. O critério de arredondamento simétrico definido consiste em:

$$ 
y_i = \begin{cases} 
1 & \text{se } a_i \geq 0.5 \\
0 & \text{se } a_i < 0.5 
\end{cases}
$$

Essa rotina foi aplicada às saídas contínuas logo após a propagação $forward$ dos dados de teste pela rede já treinada, o que garante a correspondência inteira exigida para calcular as taxas de acerto.

---

### 4. Validação da Rede (Conjunto de Teste)

Abaixo está o resultado da validação com o conjunto de teste providenciado, demonstrando a saída desejada versus a saída efetiva fornecida pela rede e pós-processada:

| Amostra | x1 | x2 | x3 | x4 | d1 | d2 | d3 | y1 | y2 | y3 | Acerto? |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.8622 | 0.7101 | 0.6236 | 0.7894 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 2 | 0.2741 | 0.1552 | 0.1333 | 0.1516 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 3 | 0.6772 | 0.8516 | 0.6543 | 0.7573 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 4 | 0.2178 | 0.5039 | 0.6415 | 0.5039 | 0 | 1 | 0 | 0 | 1 | 0 | Sim |
| 5 | 0.7260 | 0.7500 | 0.7007 | 0.4953 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 6 | 0.2473 | 0.2941 | 0.4248 | 0.3087 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 7 | 0.5682 | 0.5683 | 0.5054 | 0.4426 | 0 | 1 | 0 | 0 | 1 | 0 | Sim |
| 8 | 0.6566 | 0.6715 | 0.4952 | 0.3951 | 0 | 1 | 0 | 0 | 1 | 0 | Sim |
| 9 | 0.0705 | 0.4717 | 0.2921 | 0.2954 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 10 | 0.1187 | 0.2568 | 0.3140 | 0.3037 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 11 | 0.5673 | 0.7011 | 0.4083 | 0.5552 | 0 | 1 | 0 | 0 | 1 | 0 | Sim |
| 12 | 0.3164 | 0.2251 | 0.3526 | 0.2560 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 13 | 0.7884 | 0.9568 | 0.6825 | 0.6398 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 14 | 0.9633 | 0.7850 | 0.6777 | 0.6059 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 15 | 0.7739 | 0.8505 | 0.7934 | 0.6626 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 16 | 0.4219 | 0.4136 | 0.1408 | 0.0940 | 1 | 0 | 0 | 1 | 0 | 0 | Sim |
| 17 | 0.6616 | 0.4365 | 0.6597 | 0.8129 | 0 | 0 | 1 | 0 | 0 | 1 | Sim |
| 18 | 0.7325 | 0.4761 | 0.3888 | 0.5683 | 0 | 1 | 0 | 0 | 1 | 0 | Sim |

**Taxa de Acerto Final:** **100.00%**
Tanto o modelo padrão quanto o com Momentum atingiram 100% de taxa de acerto na classificação dos ensaios laboratoriais do conjunto de teste, evidenciando a total capacidade da rede de definir corretamente o tipo de conservante.
