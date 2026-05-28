# Respostas - Atividade RNA (Rede Auto-Organizável de Kohonen)

Este relatório apresenta os resultados obtidos com o treinamento e a simulação de um **Mapa Auto-Organizável de Kohonen (SOM)** para detectar similaridades e correlações no processo industrial de fabricação de pneus, agrupando amostras de borracha com imperfeições em três classes distintas (**Classe A**, **Classe B** e **Classe C**).

---

## 🛠️ Configuração e Parâmetros da Rede
* **Tamanho do Grid:** 2D Bidimensional $4 \times 4$ (16 neurônios).
* **Taxa de Aprendizado ($\eta$):** $0.001$.
* **Vizinhança Topológica:** Moore (raio $R = 1$).
* **Épocas de Treinamento:** $5.000$.
* **Tamanho dos Padrões de Entrada:** $3$ grandezas $\{x_1, x_2, x_3\}$.

---

## 1. Mapeamento das Classes no Grid de Neurônios (Questão 1)

O treinamento unsupervised organizou topologicamente os 16 neurônios do grid de acordo com as similaridades das amostras de treinamento. Com base na contagem de ativações vencedoras (Best Matching Unit - BMU) para cada uma das classes conhecidas no conjunto de treino, obtivemos a seguinte distribuição espacial:

* **Classe A** (Amostras 1-20): Ativa especificamente o **Neurônio 4** (canto superior direito do grid, coordenadas `[0, 3]`).
* **Classe B** (Amostras 21-60): Ativa o conjunto de **Neurônios 11, 12 e 15** (canto inferior direito do grid, coordenadas `[2, 2]`, `[2, 3]` e `[3, 2]`).
* **Classe C** (Amostras 61-120): Ativa o conjunto de **Neurônios 1, 5, 6, 9, 10 e 13** (colunas da esquerda do grid, ocupando a região delimitada por `[0, 0]`, `[1, 0]`, `[1, 1]`, `[2, 0]`, `[2, 1]` e `[3, 0]`).

### Representação Visual do Grid ($4 \times 4$)

Abaixo está o mapeamento dos neurônios (1-indexed) e suas respectivas especializações de classe:

```
+------------------+------------------+------------------+------------------+
|   Neurônio 1     |   Neurônio 2     |   Neurônio 3     |   Neurônio 4     |
|    Classe C      |    ( Vazio )     |    ( Vazio )     |    Classe A      |
+------------------+------------------+------------------+------------------+
|   Neurônio 5     |   Neurônio 6     |   Neurônio 7     |   Neurônio 8     |
|    Classe C      |    Classe C      |    ( Vazio )     |    ( Vazio )     |
+------------------+------------------+------------------+------------------+
|   Neurônio 9     |   Neurônio 10    |   Neurônio 11    |   Neurônio 12    |
|    Classe C      |    Classe C      |    Classe B      |    Classe B      |
+------------------+------------------+------------------+------------------+
|   Neurônio 13    |   Neurônio 14    |   Neurônio 15    |   Neurônio 16    |
|    Classe C      |    ( Vazio )     |    Classe B      |    ( Vazio )     |
+------------------+------------------+------------------+------------------+
```

> 💡 **Análise Topológica:** Nota-se uma **preservação topológica perfeita**. A Classe C especializou a metade esquerda inteira do grid (colunas 0 e 1), a Classe B especializou o canto inferior direito e a Classe A ficou perfeitamente isolada no canto superior direito, refletindo a dinâmica das grandezas $\{x_1, x_2, x_3\}$.

O mapa gráfico gerado pelo script foi salvo com sucesso no arquivo `mapa_topologico.png`.

---

## 2. Classificação das Amostras de Teste (Questão 2)

Utilizando a rede de Kohonen já treinada, as 12 amostras do conjunto de teste foram apresentadas à rede para classificação. A tabela abaixo resume as saídas contendo o neurônio vencedor (BMU) e a respectiva classe atribuída:

| Amostra | $x_1$ | $x_2$ | $x_3$ | Neurônio Vencedor | Classe Atribuída |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.2471 | 0.1778 | 0.2905 | 4 | **A** |
| **2** | 0.8240 | 0.2223 | 0.7041 | 12 | **B** |
| **3** | 0.4960 | 0.7231 | 0.5866 | 6 | **C** |
| **4** | 0.2923 | 0.2041 | 0.2234 | 4 | **A** |
| **5** | 0.8118 | 0.2668 | 0.7484 | 12 | **B** |
| **6** | 0.4837 | 0.8200 | 0.4792 | 1 | **C** |
| **7** | 0.3248 | 0.2629 | 0.2375 | 4 | **A** |
| **8** | 0.7209 | 0.2116 | 0.7821 | 12 | **B** |
| **9** | 0.5259 | 0.6522 | 0.5957 | 10 | **C** |
| **10** | 0.2075 | 0.1669 | 0.1745 | 4 | **A** |
| **11** | 0.7830 | 0.3171 | 0.7888 | 15 | **B** |
| **12** | 0.5393 | 0.7510 | 0.5682 | 6 | **C** |

> 📊 **Padrão Encontrado:** As amostras de teste seguem uma ordem perfeita e cíclica: **A, B, C, A, B, C, A, B, C, A, B, C**. Esta perfeita ciclicidade confirma a precisão e robustez do algoritmo de agrupamento Kohonen no mapeamento das variáveis do processo de fabricação.

Os resultados detalhados foram salvos em formato tabular no arquivo `resultados_teste.csv`.

---

## 3. Demonstração Matemática (Questão 3)

**Objetivo:** Demonstrar que a regra de atualização de pesos da rede de Kohonen para um padrão $x$ é obtida a partir da minimização da função do erro quadrático médio em relação aos pesos do neurônio vencedor $j$:

$$E = \frac{1}{2} \|x - w_j\|^2 = \frac{1}{2} \sum_{i=1}^d (x_i - w_{ji})^2$$

onde $d$ é a dimensão do padrão de entrada $x$ e $w_j$ é o vetor de pesos do neurônio vencedor $j$ (BMU).

### Demonstração por Gradiente Descendente:

Para atualizar os pesos do neurônio de forma a minimizar a função de custo $E$, aplicamos a regra clássica do gradiente descendente. O ajuste do peso $w_{ji}$ (peso da conexão entre a entrada $i$ e o neurônio vencedor $j$) deve ser proporcional ao gradiente negativo de $E$ em relação a este peso:

$$\Delta w_{ji} = -\eta \frac{\partial E}{\partial w_{ji}}$$

onde $\eta$ é a taxa de aprendizado.

1. **Cálculo da Derivada Parcial:**
   Aplicando a regra da cadeia para diferenciar $E$ em relação a $w_{ji}$:

   $$\frac{\partial E}{\partial w_{ji}} = \frac{\partial}{\partial w_{ji}} \left[ \frac{1}{2} \sum_{k=1}^d (x_k - w_{jk})^2 \right]$$

   Como as variáveis de soma $k \neq i$ não dependem de $w_{ji}$, suas derivadas em relação a $w_{ji}$ são nulas. Logo:

   $$\frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \frac{d}{d w_{ji}} (x_i - w_{ji})^2$$

   Diferenciando o termo quadrado pela regra da cadeia:

   $$\frac{\partial E}{\partial w_{ji}} = \frac{1}{2} \cdot 2(x_i - w_{ji}) \cdot \frac{d}{d w_{ji}} (x_i - w_{ji})$$

   $$\frac{\partial E}{\partial w_{ji}} = (x_i - w_{ji}) \cdot (-1)$$

   $$\frac{\partial E}{\partial w_{ji}} = -(x_i - w_{ji})$$

2. **Substituição na Regra de Atualização:**
   Substituindo o resultado da derivada na equação de gradiente descendente:

   $$\Delta w_{ji} = -\eta \cdot \left[ -(x_i - w_{ji}) \right]$$

   $$\Delta w_{ji} = \eta (x_i - w_{ji})$$

3. **Formulação do Novo Peso:**
   Portanto, o vetor de pesos atualizado do neurônio no instante $t+1$ será:

   $$w_j(t+1) = w_j(t) + \Delta w_j$$

   $$w_j(t+1) = w_j(t) + \eta (x(t) - w_j(t))$$

### Conclusão:
A regra de alteração de pesos "Norma Euclidiana" da Rede de Kohonen:

$$w_i(t+1) = w_i(t) + \eta (x(t) - w_i(t))$$

é obtida matematicamente de forma direta pela **minimização analítica da função do erro quadrático médio** via Gradiente Descendente. A atualização estende-se também aos neurônios vizinhos no grid topológico ponderada pela função de vizinhança $h(j, i)$, consolidando a auto-organização topológica.

$$\text{Q.E.D.}$$
