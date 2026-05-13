# Respostas - Atividade PMC3 (Previsão com TDNN)

### 2. Tabela de Treinamentos

Foram executados 3 treinamentos para cada uma das topologias (Rede 1, Rede 2 e Rede 3) com taxa de aprendizado = 0.1, momentum = 0.8 e precisão = $0.5 \times 10^{-6}$. Os resultados de Erro Quadrático Médio (EQM) e Épocas estão compilados abaixo:

| Treinamento | Rede 1 (p=5, N1=10) | | Rede 2 (p=10, N1=15) | | Rede 3 (p=15, N1=25) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| | **EQM** | **Épocas** | **EQM** | **Épocas** | **EQM** | **Épocas** |
| **1º (T1)** | 0.050478 | 87 | 0.051557 | 162 | 0.494515 | 2 |
| **2º (T2)** | 0.050755 | 77 | 0.052615 | 385 | 0.494517 | 2 |
| **3º (T3)** | 0.012920 | 5188 | 0.497162 | 2 | 0.494511 | 2 |

---

### 3. Validação da Rede

A tabela abaixo mostra a previsão passo-a-passo (t=101 até 120) para todos os treinamentos das três redes e suas métricas comparadas ao valor desejado (série temporal real):

| Amostra | f(t) | Rede 1 (T1) | Rede 1 (T2) | Rede 1 (T3) | Rede 2 (T1) | Rede 2 (T2) | Rede 2 (T3) | Rede 3 (T1) | Rede 3 (T2) | Rede 3 (T3) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| t=101 | 0.4173 | 0.3284 | 0.3360 | 0.6328 | 0.3231 | 0.3195 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=102 | 0.0062 | 0.3303 | 0.3231 | 0.0665 | 0.3262 | 0.3229 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=103 | 0.3387 | 0.3400 | 0.3394 | 0.5470 | 0.3277 | 0.3373 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=104 | 0.1886 | 0.3419 | 0.3152 | 0.2772 | 0.3351 | 0.3279 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=105 | 0.7418 | 0.3419 | 0.3402 | 0.6030 | 0.3370 | 0.3321 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=106 | 0.3138 | 0.3284 | 0.3235 | 0.2836 | 0.3224 | 0.3220 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=107 | 0.4466 | 0.3218 | 0.3345 | 0.4869 | 0.3300 | 0.3240 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=108 | 0.0835 | 0.3214 | 0.3250 | 0.0883 | 0.3213 | 0.3182 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=109 | 0.1930 | 0.3301 | 0.3342 | 0.2590 | 0.3252 | 0.3359 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=110 | 0.3807 | 0.3395 | 0.3190 | 0.2191 | 0.3330 | 0.3245 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=111 | 0.5438 | 0.3393 | 0.3312 | 0.4792 | 0.3311 | 0.3312 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=112 | 0.5897 | 0.3298 | 0.3233 | 0.4923 | 0.3220 | 0.3174 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=113 | 0.3536 | 0.3188 | 0.3313 | 0.2569 | 0.3218 | 0.3207 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=114 | 0.2210 | 0.3184 | 0.3291 | 0.1635 | 0.3157 | 0.3160 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=115 | 0.0631 | 0.3256 | 0.3272 | 0.1261 | 0.3204 | 0.3268 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=116 | 0.4499 | 0.3381 | 0.3257 | 0.2574 | 0.3261 | 0.3261 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=117 | 0.2564 | 0.3392 | 0.3234 | 0.3511 | 0.3265 | 0.3293 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=118 | 0.7642 | 0.3361 | 0.3275 | 0.6448 | 0.3263 | 0.3186 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=119 | 0.1411 | 0.3228 | 0.3280 | 0.1805 | 0.3200 | 0.3192 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| t=120 | 0.3626 | 0.3228 | 0.3336 | 0.4234 | 0.3178 | 0.3177 | 0.9999 | 0.9999 | 0.9999 | 0.9999 |
| **Erro Relativo Médio** | - | 3.2945 | 3.2346 | **0.7877** | 3.2549 | 3.2349 | 11.1247 | 11.1255 | 11.1255 | 11.1255 |
| **Variância** | - | 127.29 | 121.70 | **4.2633** | 124.06 | 121.49 | 1183.65 | 1183.80 | 1183.81 | 1183.80 |

---

### 4. Gráficos de Erro Quadrático Médio

Abaixo, encontram-se os gráficos de EQM em função da época do **melhor treinamento** de cada rede (T3 para a Rede 1, T1 para a Rede 2 e T3 para a Rede 3).

![Gráficos EQM Topologias](file:///c:/Users/isabe/Desktop/LabInteligenciaArtificial/PMC3/graficos_eqm_topologias.png)

---

### 5. Gráficos de Valores Desejados vs Estimados

Abaixo estão as sobreposições entre a curva real ($f(t)$) e a curva estimada pela topologia (com a melhor configuração de treinamento de cada uma) no domínio de teste $t \in [101, 120]$.

![Gráficos Estimativas Topologias](file:///c:/Users/isabe/Desktop/LabInteligenciaArtificial/PMC3/graficos_estimativas_topologias.png)

---

### 6. Indicação da Melhor Configuração

A configuração mais adequada é a **Rede 1 (p=5, N1=10)** utilizando o **Treinamento T3**.

**Justificativa:** 
Ao analisarmos os dados, as Redes 2 e 3 superparametrizaram o problema (excesso de neurônios ocultos e janelas de entradas grandes demais para a amostragem) e rapidamente travaram em mínimos locais de altíssimo erro (saturando a saída em `~0.9999`). 
Já a Rede 1 apresentou a melhor generalização e convergência. Especialmente o treinamento T3 escapou dos mínimos locais rasos em que T1 e T2 ficaram presos, treinando até a época 5188 e obtendo não só o menor EQM de validação final ($0.012$), mas disparadamente o **menor Erro Relativo Médio (0.78 contra > 3.2 dos demais)** e a menor variância (apenas 4.2 contra > 120 dos demais), sendo o único capaz de mapear a dinâmica da série temporal adequadamente.

---

### 7. Investigação sobre Algoritmos Variantes

**a. Algoritmo de treinamento Resilient-Propagation (RProp)**
O RProp é uma adaptação robusta do algoritmo de gradiente descente projetada especialmente para superar o problema de *vanishing gradient* (gradientes que tendem a zero) frequentemente causados pelas funções sigmoides.
*   **Características Principais:** A principal diferença do RProp é que ele desconsidera a **magnitude** da derivada parcial e utiliza apenas o seu **sinal** para guiar a atualização dos pesos. Se o sinal do gradiente da época atual for igual ao da época anterior, o passo de atualização é acelerado; se trocar de sinal, significa que ele pulou o mínimo local, e o passo é imediatamente reduzido.
*   **Vantagens:** Acelera exponencialmente a velocidade de convergência em locais de platôs da superfície de erro e não precisa que o projetista fique ajustando a "taxa de aprendizado", pois a atualização dos passos é auto-adaptativa.

**b. Algoritmo de treinamento Levenberg-Marquardt (LM)**
O LM é um dos algoritmos de otimização de segunda-ordem mais poderosos usados em Redes Neurais, criado para aproximar o método de Newton de forma mais computacionalmente acessível utilizando a matriz Jacobiana em vez do cálculo direto (e caríssimo) da matriz Hessiana.
*   **Características Principais:** Ele atua como um algoritmo híbrido (interpolação): quando se está distante do erro mínimo, ele se comporta como o gradiente descendente convencional; quando se aproxima do vale do mínimo do erro, ele muda suavemente para o método de Gauss-Newton, que é muito mais exato e rápido em curvaturas acentuadas.
*   **Vantagens:** Possui a convergência de treinamento **mais rápida existente** para redes neurais de pequeno ou médio porte na grande maioria dos cenários não-lineares. É excelente para evitar travamentos em superfícies de erro difíceis.
*   **Desvantagens:** É devorador de recursos de memória computacional (computar a Matriz Jacobiana de grandes redes consome um tempo proibitivo em backpropagation), por isso só é viável em redes TDNN com poucas dezenas de neurônios (como as abordadas no experimento).
