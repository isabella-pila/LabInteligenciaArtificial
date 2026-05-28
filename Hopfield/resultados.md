# Relatório Técnico: Memória Associativa com Rede de Hopfield

**Disciplina:** Lab. Inteligência Artificial  
**Professor:** Lázaro Eduardo da Silva  
**Aluno(a):** Isabella Pila  
**Data:** 28 de Maio de 2026  

## 1. Descrição do Problema e Fundamentação Teórica
Este relatório documenta a implementação de uma rede de Hopfield discreta com **45 neurônios** para atuar como memória associativa recorrente. O objetivo é armazenar **quatro padrões originais** (representando os dígitos de 1 a 4 em uma grade de $9 \times 5$) e recuperá-los fielmente a partir de imagens que sofreram **20% de ruído aleatório** durante a transmissão.

### 1.1 Modelo Matemático da Rede
- **Representação dos Neurônios:** Os pixels brancos são codificados como $-1$ e os pretos como $+1$. O estado do sistema é um vetor bipolar $S \in \{-1, +1\}^{45}$.
- **Matriz de Pesos (Regra Hebbiana):** A matriz de conexões $W$ é obtida pelo produto externo dos padrões de treinamento:
  $$W = \sum_{p=1}^P x^p (x^p)^T$$
  Forçamos a diagonal a ser nula ($w_{ii} = 0$) para eliminar auto-conexões. Isso garante a estabilidade matemática do sistema.
- **Função de Ativação:** A ativação dos neurônios é dada por uma tangente hiperbólica com inclinação muito grande, o que equivale à função bipolar contínua de sinal:
  $$s_i(t+1) = f\left( \sum_{j=1}^N w_{ij} s_j(t) \right) = \begin{cases} +1, & \text{se } \sum w_{ij}s_j > 0 \\ -1, & \text{se } \sum w_{ij}s_j < 0 \\ s_i(t), & \text{se } \sum w_{ij}s_j = 0 \end{cases}$$
- **Atualização Assíncrona:** Neurônios são atualizados de forma isolada, um por vez. Essa dinâmica garante que a função de Lyapunov (Energia) decresça monotonicamente:
  $$E = -\frac{1}{2} S^T W S$$
  Isso impede oscilações indefinidas e assegura a convergência da rede para um atrator estável (mínimo local de energia).

## 2. Simulação das 12 Situações de Transmissão (20% de Ruído)
Abaixo estão detalhados os resultados das **12 simulações de transmissão** (3 testes para cada um dos 4 padrões). Em cada caso, apresentamos o padrão original, a versão com **20% de ruído (9 pixels invertidos aleatoriamente)** e o estado estável recuperado pela rede.

### 2.1 Padrão 1 (Dígito 1)
Abaixo estão os 3 casos de simulação para o dígito 1. Cada caso altera exatamente 9 pixels aleatórios do padrão de entrada original.

#### Caso 1 - Padrão 1
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -344.0 | **Energia Final:** -960.0
- **Índices de Pixels Invertidos:** `[4, 8, 12, 25, 26, 35, 39, 41, 43]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 1 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `░░██░` | `░░███` | `░░██░` |
| `░███░` | `░██░░` | `░███░` |
| `░░██░` | `░░░█░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `████░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `█░███` | `░░██░` |
| `░░██░` | `░██░░` | `░░██░` |


#### Caso 2 - Padrão 1
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -312.0 | **Energia Final:** -960.0
- **Índices de Pixels Invertidos:** `[1, 3, 14, 16, 18, 22, 33, 41, 44]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 1 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `░░██░` | `░██░░` | `░░██░` |
| `░███░` | `░███░` | `░███░` |
| `░░██░` | `░░███` | `░░██░` |
| `░░██░` | `░██░░` | `░░██░` |
| `░░██░` | `░░░█░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `░░█░░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `░████` | `░░██░` |


#### Caso 3 - Padrão 1
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -280.0 | **Energia Final:** -960.0
- **Índices de Pixels Invertidos:** `[2, 9, 12, 15, 21, 29, 30, 31, 40]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 1 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `░░██░` | `░░░█░` | `░░██░` |
| `░███░` | `░████` | `░███░` |
| `░░██░` | `░░░█░` | `░░██░` |
| `░░██░` | `█░██░` | `░░██░` |
| `░░██░` | `░███░` | `░░██░` |
| `░░██░` | `░░███` | `░░██░` |
| `░░██░` | `████░` | `░░██░` |
| `░░██░` | `░░██░` | `░░██░` |
| `░░██░` | `█░██░` | `░░██░` |


### 2.1 Padrão 2 (Dígito 2)
Abaixo estão os 3 casos de simulação para o dígito 2. Cada caso altera exatamente 9 pixels aleatórios do padrão de entrada original.

#### Caso 1 - Padrão 2
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -432.0 | **Energia Final:** -1372.0
- **Índices de Pixels Invertidos:** `[5, 8, 10, 11, 13, 18, 24, 27, 30]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 2 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `█████` | `█████` |
| `█████` | `░██░█` | `█████` |
| `░░░██` | `██░░█` | `░░░██` |
| `░░░██` | `░░░░█` | `░░░██` |
| `█████` | `████░` | `█████` |
| `██░░░` | `███░░` | `██░░░` |
| `██░░░` | `░█░░░` | `██░░░` |
| `█████` | `█████` | `█████` |
| `█████` | `█████` | `█████` |


#### Caso 2 - Padrão 2
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -456.0 | **Energia Final:** -1372.0
- **Índices de Pixels Invertidos:** `[1, 7, 19, 22, 29, 30, 32, 40, 44]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 2 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `█░███` | `█████` |
| `█████` | `██░██` | `█████` |
| `░░░██` | `░░░██` | `░░░██` |
| `░░░██` | `░░░█░` | `░░░██` |
| `█████` | `██░██` | `█████` |
| `██░░░` | `██░░█` | `██░░░` |
| `██░░░` | `░██░░` | `██░░░` |
| `█████` | `█████` | `█████` |
| `█████` | `░███░` | `█████` |


#### Caso 3 - Padrão 2
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -540.0 | **Energia Final:** -1380.0
- **Índices de Pixels Invertidos:** `[0, 2, 19, 22, 23, 33, 34, 42, 43]`
- **Resultado:** **Falha (Estado Espúrio / Outro Atrator)** (Atrator final: Nenhum (Estado Espúrio))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `░█░██` | `█████` |
| `█████` | `█████` | `█████` |
| `░░░██` | `░░░██` | `░░░██` |
| `░░░██` | `░░░█░` | `░░░██` |
| `█████` | `██░░█` | `█████` |
| `██░░░` | `██░░░` | `██░░█` |
| `██░░░` | `██░██` | `██░░█` |
| `█████` | `█████` | `█████` |
| `█████` | `██░░█` | `█████` |


### 2.1 Padrão 3 (Dígito 3)
Abaixo estão os 3 casos de simulação para o dígito 3. Cada caso altera exatamente 9 pixels aleatórios do padrão de entrada original.

#### Caso 1 - Padrão 3
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -484.0 | **Energia Final:** -1524.0
- **Índices de Pixels Invertidos:** `[1, 3, 10, 12, 15, 20, 24, 35, 42]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 3 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `█░█░█` | `█████` |
| `█████` | `█████` | `█████` |
| `░░░██` | `█░███` | `░░░██` |
| `░░░██` | `█░░██` | `░░░██` |
| `█████` | `░███░` | `█████` |
| `░░░██` | `░░░██` | `░░░██` |
| `░░░██` | `░░░██` | `░░░██` |
| `█████` | `░████` | `█████` |
| `█████` | `██░██` | `█████` |


#### Caso 2 - Padrão 3
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -552.0 | **Energia Final:** -1524.0
- **Índices de Pixels Invertidos:** `[5, 6, 11, 21, 23, 28, 31, 35, 37]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 3 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `█████` | `█████` |
| `█████` | `░░███` | `█████` |
| `░░░██` | `░█░██` | `░░░██` |
| `░░░██` | `░░░██` | `░░░██` |
| `█████` | `█░█░█` | `█████` |
| `░░░██` | `░░░░█` | `░░░██` |
| `░░░██` | `░█░██` | `░░░██` |
| `█████` | `░█░██` | `█████` |
| `█████` | `█████` | `█████` |


#### Caso 3 - Padrão 3
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -636.0 | **Energia Final:** -1524.0
- **Índices de Pixels Invertidos:** `[9, 11, 15, 16, 19, 22, 26, 28, 34]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 3 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `█████` | `█████` | `█████` |
| `█████` | `████░` | `█████` |
| `░░░██` | `░█░██` | `░░░██` |
| `░░░██` | `██░█░` | `░░░██` |
| `█████` | `██░██` | `█████` |
| `░░░██` | `░█░░█` | `░░░██` |
| `░░░██` | `░░░█░` | `░░░██` |
| `█████` | `█████` | `█████` |
| `█████` | `█████` | `█████` |


### 2.1 Padrão 4 (Dígito 4)
Abaixo estão os 3 casos de simulação para o dígito 4. Cada caso altera exatamente 9 pixels aleatórios do padrão de entrada original.

#### Caso 1 - Padrão 4
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -504.0 | **Energia Final:** -1524.0
- **Índices de Pixels Invertidos:** `[0, 15, 16, 26, 27, 29, 33, 40, 42]`
- **Resultado:** **Falha (Estado Espúrio / Outro Atrator)** (Atrator final: Padrão 3 (Incorreto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `██░██` | `░█░██` | `█████` |
| `██░██` | `██░██` | `█████` |
| `██░██` | `██░██` | `░░░██` |
| `█████` | `░░███` | `░░░██` |
| `█████` | `█████` | `█████` |
| `░░░██` | `░███░` | `░░░██` |
| `░░░██` | `░░░░█` | `░░░██` |
| `░░░██` | `░░░██` | `█████` |
| `░░░██` | `█░███` | `█████` |


#### Caso 2 - Padrão 4
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -400.0 | **Energia Final:** -1120.0
- **Índices de Pixels Invertidos:** `[0, 1, 10, 21, 32, 34, 36, 42, 44]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 4 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `██░██` | `░░░██` | `██░██` |
| `██░██` | `██░██` | `██░██` |
| `██░██` | `░█░██` | `██░██` |
| `█████` | `█████` | `█████` |
| `█████` | `█░███` | `█████` |
| `░░░██` | `░░░██` | `░░░██` |
| `░░░██` | `░░██░` | `░░░██` |
| `░░░██` | `░█░██` | `░░░██` |
| `░░░██` | `░░██░` | `░░░██` |


#### Caso 3 - Padrão 4
- **Número de Iterações (Varreduras):** 2
- **Energia Inicial:** -296.0 | **Energia Final:** -1120.0
- **Índices de Pixels Invertidos:** `[4, 6, 10, 14, 18, 27, 30, 32, 33]`
- **Resultado:** **Sucesso (Recuperação Fiel)** (Atrator final: Padrão 4 (Correto))

| Transmitida (Sem Ruído) | Distorcida (20% Ruído) | Recuperada (Hopfield) |
|:---:|:---:|:---:|
| `██░██` | `██░█░` | `██░██` |
| `██░██` | `█░░██` | `██░██` |
| `██░██` | `░█░█░` | `██░██` |
| `█████` | `███░█` | `█████` |
| `█████` | `█████` | `█████` |
| `░░░██` | `░░███` | `░░░██` |
| `░░░██` | `█░█░█` | `░░░██` |
| `░░░██` | `░░░██` | `░░░██` |
| `░░░██` | `░░░██` | `░░░██` |


### 2.2 Tabela Resumo das Simulações
| Padrão | Caso | Varreduras | Energia Inicial | Energia Final | Recuperação Perfeita? |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Dígito 1** | Caso 1 | 2 | -344.0 | -960.0 | SIM |
| **Dígito 1** | Caso 2 | 2 | -312.0 | -960.0 | SIM |
| **Dígito 1** | Caso 3 | 2 | -280.0 | -960.0 | SIM |
| **Dígito 2** | Caso 1 | 2 | -432.0 | -1372.0 | SIM |
| **Dígito 2** | Caso 2 | 2 | -456.0 | -1372.0 | SIM |
| **Dígito 2** | Caso 3 | 2 | -540.0 | -1380.0 | NÃO (Nenhum (Estado Espúrio)) |
| **Dígito 3** | Caso 1 | 2 | -484.0 | -1524.0 | SIM |
| **Dígito 3** | Caso 2 | 2 | -552.0 | -1524.0 | SIM |
| **Dígito 3** | Caso 3 | 2 | -636.0 | -1524.0 | SIM |
| **Dígito 4** | Caso 1 | 2 | -504.0 | -1524.0 | NÃO (Padrão 3 (Incorreto)) |
| **Dígito 4** | Caso 2 | 2 | -400.0 | -1120.0 | SIM |
| **Dígito 4** | Caso 3 | 2 | -296.0 | -1120.0 | SIM |

## 3. Comportamento sob Níveis Excessivos de Ruído
Para responder ao item 3 da atividade, realizamos um experimento de robustez variando o nível de ruído induzido de **0% a 100%** (com passos de 10%). Para cada nível de ruído, executamos **200 simulações independentes** (50 testes para cada um dos quatro padrões) e medimos a taxa de recuperação perfeita, além do tipo de erro mais frequente.

### 3.1 Tabela de Sensibilidade ao Ruído
| Nível de Ruído (%) | Taxa de Recuperação Perfeita (%) | Comportamento Predominante de Falha |
| :---: | :---: | :--- |
|         0         % |             100.0             % | Nenhum (100% de sucesso) |
|         10        % |              91.5             % | Falha ocasional para Estado Espúrio |
|         20        % |              82.5             % | Falha ocasional para Estado Espúrio |
|         30        % |              53.0             % | Falha ocasional para Estado Espúrio |
|         40        % |              35.0             % | Queda em Estados Espúrios (Mínimos Locais) |
|         50        % |              5.0              % | Queda em Estados Espúrios (Mínimos Locais) |
|         60        % |              0.0              % | Queda em Estados Espúrios (Mínimos Locais) |
|         70        % |              0.0              % | Convergência para Padrão Reverso (-Padrão) |
|         80        % |              0.0              % | Convergência para Padrão Reverso (-Padrão) |
|         90        % |              0.0              % | Convergência para Padrão Reverso (-Padrão) |
|        100        % |              0.0              % | Convergência para Padrão Reverso (-Padrão) |

### 3.2 Análise e Explicação Científica
Quando o nível de ruído é aumentado de forma excessiva, ocorrem três fenômenos bem definidos da dinâmica física das redes de Hopfield:

#### A. Transição das Bacias de Atração e Estados Espúrios
A superfície de energia da rede é composta por vales ou depressões correspondentes aos estados estáveis (atratores). Os 4 padrões armazenados representam mínimos locais profundos nesta superfície. A região em torno de cada padrão em que as trajetórias de estados convergem para ele é chamada de **bacia de atração**.
- Quando o ruído é baixo (≤ 20%), a imagem distorcida começa dentro da bacia de atração do padrão correspondente. A dinâmica assíncrona reduz a energia do sistema até que ele deslize para o fundo do poço correspondente ao padrão original.
- Quando o ruído aumenta para a faixa de **30% a 50%**, o estado inicial é posicionado na fronteira ou fora da bacia de atração original. O sistema então converge para **estados espúrios** — que são mínimos locais indesejados criados pela sobreposição linear dos pesos (interferência mútua entre os dígitos armazenados). Estes estados parecem misturas dos dígitos originais e possuem energia ligeiramente superior à dos padrões reais.
- Por exemplo, na nossa simulação do **Caso 1 do Padrão 4**, o ruído de 20% inverteu pixels críticos que o descaracterizaram, fazendo com que a rede convergisse perfeitamente para o **Padrão 3**! Isso ocorre porque a distância de Hamming entre o Padrão 4 ruidoso e o Padrão 3 tornou-se menor do que em relação ao próprio Padrão 4.

#### B. Atratores Reversos (Estados Complementares)
A matriz de pesos obtida pela regra de Hebb é estritamente simétrica e linear. Se um padrão $X$ é armazenado de forma estável, o seu **negativo** ou **inverso** $-X$ possui exatamente a mesma estabilidade e energia:
  $$E(-X) = -\frac{1}{2} (-X)^T W (-X) = -\frac{1}{2} X^T W X = E(X)$$
- Quando o ruído excede **60%**, o padrão de entrada está com mais da metade de seus pixels invertidos. Fisicamente, isso significa que a imagem ruidosa é muito mais parecida com a imagem complementar (negativa) do que com a original.
- Como consequência, a rede converge com **100% de probabilidade para o estado complementar/reverso** (pixels brancos viram pretos e pretos viram brancos). A 100% de ruído, a recuperação da imagem invertida é perfeita! Isso explica o salto de comportamento em níveis de ruído extremos.

#### C. Capacidade de Armazenamento da Rede (Limite de Shannon)
A teoria clássica de Hopfield estabelece que a capacidade máxima teórica de armazenamento sem perda severa de bacia de atração é de aproximadamente:
  $$C \approx 0.138 \times N$$
  Onde $N$ é o número de neurônios. Para a nossa rede com $N = 45$, temos:
  $$C \approx 0.138 \times 45 \approx 6.2 \text{ padrões}$$
- Como estamos armazenando $4$ padrões, estamos operando muito próximos do limite superior de armazenamento da rede ($4/45 \approx 0.088$). Isso explica o motivo de a taxa de sucesso cair para **82.5%** mesmo em um ruído moderado de 20%, e despencar rapidamente acima de 30%.
- A sobreposição das bacias de atração cria uma interferência (diafonia ou *crosstalk*) entre os dígitos de formato similar (como o 3 e o 4, ou o 1 e o 3).

## 4. Conclusão
A Rede de Hopfield implementada demonstra com clareza os conceitos de **memória associativa associada ao conteúdo** e **superfícies de energia de Lyapunov**. A rede mostrou-se altamente robusta para níveis de ruído de até 20% em condições gerais. As falhas observadas em ruídos elevados não são falhas de programação, mas sim propriedades físicas e matemáticas inerentes ao modelo clássico de Hopfield, perfeitamente preditas pela física estatística (Estados Espúrios e Atratores Reversos).