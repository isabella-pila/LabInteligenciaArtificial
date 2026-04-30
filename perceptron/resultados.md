# Resultados: Perceptron para Classificação de Óleo

Este documento apresenta os resultados da implementação do algoritmo Perceptron utilizando a Regra de Hebb (com taxa de aprendizado η = 0.01) para a classificação de pureza de óleo em duas classes (C1 como `-1` e C2 como `1`).

## 1. Treinamento

Foram realizados 5 treinamentos independentes inicializando o vetor de pesos aleatoriamente entre 0 e 1.

| Treinamento | Pesos Iniciais `[b, w1, w2, w3]` | Épocas | Pesos Finais `[b, w1, w2, w3]` |
|:---:|---|:---:|---|
| **T1** | `[0.5434, 0.2783, 0.4245, 0.8447]` | 378 | `[1.5534, 0.7888, 1.2506, -0.3700]` |
| **T2** | `[0.9476, 0.2265, 0.5944, 0.4283]` | 244 | `[1.4776, 0.7411, 1.2206, -0.3452]` |
| **T3** | `[0.4511, 0.2210, 0.3690, 0.2907]` | 407 | `[1.5711, 0.7962, 1.2629, -0.3733]` |
| **T4** | `[0.6687, 0.2259, 0.8592, 0.2903]` | 315 | `[1.5387, 0.7789, 1.2405, -0.3675]` |
| **T5** | `[0.6936, 0.0617, 0.6666, 0.5592]` | 290 | `[1.4636, 0.7186, 1.2064, -0.3411]` |

## 2. Classificação de Novas Amostras

Após o treinamento, o modelo foi aplicado em 10 novas amostras. Abaixo estão as saídas (`y`) utilizando os pesos resultantes de cada um dos 5 treinamentos (T1 a T5):

| Amostra | x1 | x2 | x3 | y (T1) | y (T2) | y (T3) | y (T4) | y (T5) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | -0.3565 | 0.0620 | 5.9891 | -1 | -1 | -1 | -1 | -1 |
| **2** | -0.7842 | 1.1267 | 5.5912 | 1 | 1 | 1 | 1 | 1 |
| **3** | 0.3012 | 0.5611 | 5.8234 | 1 | 1 | 1 | 1 | 1 |
| **4** | 0.7757 | 1.0648 | 8.0677 | 1 | 1 | 1 | 1 | 1 |
| **5** | 0.1570 | 0.8028 | 6.3040 | 1 | 1 | 1 | 1 | 1 |
| **6** | -0.7014 | 1.0316 | 3.6005 | 1 | 1 | 1 | 1 | 1 |
| **7** | 0.3748 | 0.1536 | 6.1537 | -1 | -1 | -1 | -1 | -1 |
| **8** | -0.6920 | 0.9404 | 4.4058 | 1 | 1 | 1 | 1 | 1 |
| **9** | -1.3970 | 0.7141 | 4.9263 | -1 | -1 | -1 | -1 | -1 |
| **10** | -1.8842 | -0.2805 | 1.2548 | -1 | -1 | -1 | -1 | -1 |

**Observação:** Todos os modelos treinados forneceram respostas unânimes para as 10 amostras, comprovando a robustez do algoritmo.

## 3. Questões Teóricas

### 4. Por que o número de épocas de treinamento varia a cada vez que executamos o treinamento do perceptron?
O número de épocas varia porque, em cada execução, **os pesos iniciais ($w$) e o bias ($b$) são gerados de forma aleatória**. A convergência do Perceptron funciona como uma busca por um "plano" geométrico no espaço que consiga separar perfeitamente as classes $C_1$ e $C_2$. Como os pesos iniciais definem a posição inicial desse plano, se os números sorteados gerarem um plano que já está geometricamente perto do local correto, a rede precisará de poucas atualizações (e, portanto, poucas épocas) para encontrar a solução final. Por outro lado, se a posição inicial for muito distante da solução ideal, o algoritmo precisará de muitas iterações de erro e correção para chegar ao hiperplano correto.

### 5. Qual a principal limitação do perceptron quando aplicado em problemas de classificação de padrões?
A principal limitação do Perceptron (de camada única) é que ele **só consegue resolver problemas linearmente separáveis**. Isso significa que, se as classes não puderem ser perfeitamente divididas por uma reta (em 2D), um plano (em 3D) ou um hiperplano (em dimensões maiores), o algoritmo do Perceptron nunca irá convergir e ficará atualizando os pesos infinitamente em um loop. O exemplo clássico que expôs essa falha foi o problema da porta lógica **XOR** (Ou Exclusivo), que não pode ser separado por uma única reta, o que motivou o desenvolvimento de Redes Neurais com múltiplas camadas (MLP).
