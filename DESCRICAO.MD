# Projeto 2: Clustering de Dados e Detecção de Comunidades em Grafos

## 1. Objetivo

O objetivo deste projeto é estudar a relação entre **agrupamento de dados (clustering)** e **detecção de comunidades em grafos**.

A ideia é representar o mesmo conjunto de dados de duas formas diferentes:

1. Como pontos em um espaço de atributos, utilizando um algoritmo tradicional de clustering;
2. Como um grafo de similaridade, utilizando um algoritmo de detecção de comunidades.

Os resultados das duas abordagens deverão ser comparados por meio de métricas quantitativas e visualizações.

---

## 2. Formulação do problema

Considere um conjunto de dados formado por $n$ objetos:

$$
X = \{x_1, x_2, \ldots, x_n\}
$$

Cada objeto $x_i$ é representado por $d$ atributos:

$$
x_i = (x_{i1}, x_{i2}, \ldots, x_{id})
$$

O projeto deverá analisar duas abordagens.

### Abordagem 1 — Clustering tradicional

$$
X \rightarrow C_{\text{cluster}}
$$

### Abordagem 2 — Grafo e detecção de comunidades

$$
X \rightarrow G = (V, E) \rightarrow C_{\text{community}}
$$

O objetivo principal é investigar se os grupos encontrados pelas duas abordagens são semelhantes.

---

## 3. Etapa 1 — Preparação dos dados

Cada grupo deverá escolher um dataset público contendo objetos descritos por atributos numéricos.

Sugestões:

- Iris;
- Wine;
- Seeds;
- Breast Cancer;
- Outro dataset escolhido pelos alunos.

Quando necessário, os dados deverão ser normalizados ou padronizados.

Uma possibilidade é utilizar a padronização:

$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

onde:

- $\mu_j$ é a média do atributo $j$;
- $\sigma_j$ é o desvio-padrão do atributo $j$.

---

## 4. Etapa 2 — Clustering tradicional

Os alunos deverão aplicar um algoritmo tradicional de agrupamento. O método recomendado é o **K-Means**.

O objetivo do K-Means é minimizar a soma das distâncias quadráticas entre cada objeto e o centroide do seu cluster:

$$
J = \sum_{k=1}^{K} \sum_{x_i \in C_k} \|x_i - \mu_k\|^2
$$

O centroide do cluster $C_k$ é calculado por:

$$
\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
$$

Os alunos deverão justificar a escolha do número de clusters $K$.

Podem ser utilizados, por exemplo:

- Método do cotovelo (**Elbow Method**);
- **Silhouette Score**;
- Conhecimento prévio sobre o dataset.

---

## 5. Etapa 3 — Construção do grafo de similaridade

Os mesmos objetos deverão ser representados por um grafo:

$$
G = (V, E)
$$

Cada vértice $v_i \in V$ representa um objeto $x_i$.

A distância entre dois objetos poderá ser calculada utilizando a distância Euclidiana:

$$
d(x_i, x_j) =
\sqrt{
\sum_{l=1}^{d}
(x_{il} - x_{jl})^2
}
$$

A partir dessas distâncias deverá ser construído um **k-Nearest Neighbors Graph (k-NN Graph)**.

Cada objeto será conectado aos seus $k$ vizinhos mais próximos.

---

## 6. Pesos das arestas

Opcionalmente, o grafo poderá ser ponderado.

Uma função de similaridade Gaussiana pode ser utilizada:

$$
w_{ij} =
\exp\left(
-\frac{d(x_i,x_j)^2}{2\sigma^2}
\right)
$$

Dessa forma, objetos muito semelhantes terão pesos de conexão maiores.

---

## 7. Etapa 4 — Detecção de comunidades

Após a construção do grafo, deverá ser aplicado pelo menos um algoritmo de detecção de comunidades.

Sugestões:

- **Louvain**;
- **Girvan–Newman**;
- **Label Propagation**.

O resultado será uma partição dos vértices:

$$
C_{\text{community}} =
\{C_1, C_2, \ldots, C_m\}
$$

Os alunos deverão determinar o número de comunidades e analisar a composição e o tamanho de cada comunidade.

---

## 8. Avaliação por modularidade

A qualidade da estrutura de comunidades deverá ser avaliada utilizando a **modularidade**.

A modularidade é definida por:

$$
Q =
\frac{1}{2m}
\sum_{i,j}
\left(
A_{ij} -
\frac{k_i k_j}{2m}
\right)
\delta(c_i,c_j)
$$

onde:

- $A_{ij}$ representa a conexão entre os vértices $i$ e $j$;
- $k_i$ e $k_j$ são os graus dos vértices;
- $m$ é o número total de arestas;
- $c_i$ representa a comunidade do vértice $i$;
- $\delta(c_i,c_j) = 1$ quando os vértices pertencem à mesma comunidade;
- $\delta(c_i,c_j) = 0$ caso contrário.

Em geral, valores maiores de $Q$ indicam uma estrutura de comunidades mais bem definida.

---

## 9. Etapa 5 — Comparação entre clustering e detecção de comunidades

Esta é a principal etapa do projeto.

Os alunos deverão comparar a partição obtida pelo clustering:

$$
C_{\text{cluster}}
$$

com a partição obtida pela detecção de comunidades:

$$
C_{\text{community}}
$$

Uma das métricas recomendadas é o **Adjusted Rand Index (ARI)**:

$$
ARI =
\frac{
\displaystyle
\sum_{ij}\binom{n_{ij}}{2}
-
\frac{
\displaystyle
\sum_i\binom{a_i}{2}
\sum_j\binom{b_j}{2}
}{
\displaystyle
\binom{n}{2}
}
}{
\displaystyle
\frac{1}{2}
\left[
\sum_i\binom{a_i}{2}
+
\sum_j\binom{b_j}{2}
\right]
-
\frac{
\displaystyle
\sum_i\binom{a_i}{2}
\sum_j\binom{b_j}{2}
}{
\displaystyle
\binom{n}{2}
}
}
$$

onde:

- $n$ = número total de objetos;
- $n_{ij}$ = número de objetos que pertencem simultaneamente ao cluster $i$ da primeira partição e ao cluster $j$ da segunda partição;
- $a_i = \sum_j n_{ij}$ = tamanho do cluster $i$ da primeira partição;
- $b_j = \sum_i n_{ij}$ = tamanho do cluster $j$ da segunda partição;
- $\binom{n}{2}$ = número total de pares de objetos.

### Interpretação do ARI

- **ARI = 1** → as duas partições são idênticas;
- **ARI ≈ 0** → a concordância é aproximadamente a esperada ao acaso;
- **ARI < 0** → a concordância é inferior à esperada ao acaso.

Outra possibilidade é utilizar o **Normalized Mutual Information (NMI)**:

$$
NMI(U,V) =
\frac{2I(U;V)}{H(U)+H(V)}
$$

onde:

- $I(U;V)$ representa a informação mútua entre as duas partições;
- $H(U)$ e $H(V)$ representam suas respectivas entropias.

A fórmula da **Informação Mútua (Mutual Information — MI)** entre duas partições $U$ e $V$ é:

$$
I(U;V) =
\sum_{i=1}^{|U|}
\sum_{j=1}^{|V|}
p(u_i,v_j)
\log
\left(
\frac{p(u_i,v_j)}
{p(u_i)p(v_j)}
\right)
$$

onde:

- $U = \{u_1, u_2, \ldots, u_{|U|}\}$ representa a primeira partição;
- $V = \{v_1, v_2, \ldots, v_{|V|}\}$ representa a segunda partição;
- $p(u_i,v_j)$ é a probabilidade de um objeto pertencer simultaneamente aos grupos $u_i$ e $v_j$;
- $p(u_i)$ é a probabilidade de um objeto pertencer ao grupo $u_i$;
- $p(v_j)$ é a probabilidade de um objeto pertencer ao grupo $v_j$.

Os alunos deverão utilizar **pelo menos uma dessas métricas** para avaliar a similaridade entre os dois agrupamentos.

---

## 10. Experimento com diferentes valores de $k$

Os alunos deverão investigar como o valor de $k$ utilizado na construção do grafo influencia o resultado da detecção de comunidades.

Para cada valor de $k$, deverão registrar:

| $k$ | Nº de comunidades | Modularidade $Q$ | NMI/ARI |
|---:|---:|---:|---:|
| 3  | — | — | — |
| 5  | — | — | — |
| 10 | — | — | — |
| 15 | — | — | — |
| 20 | — | — | — |

Os resultados deverão ser apresentados também por meio de gráficos.

Os alunos deverão discutir:

- Como o aumento de $k$ altera a densidade do grafo?
- Como o número de comunidades muda?
- Qual valor de $k$ produz os melhores resultados?
- O que acontece quando $k$ é muito pequeno?
- O que acontece quando $k$ é muito grande?

---

## 11. Comparação com os rótulos reais

Quando o dataset possuir classes ou rótulos conhecidos, os alunos deverão comparar os resultados obtidos com o chamado **ground truth**.

Deverão ser realizadas, quando possível, três comparações:

1. **K-Means × Ground Truth**
2. **Community Detection × Ground Truth**
3. **K-Means × Community Detection**

Poderão ser utilizadas métricas como:

- ARI;
- NMI;
- Silhouette Score.

O objetivo é verificar se a estrutura encontrada pelos algoritmos corresponde às classes conhecidas dos dados.

---

## 12. Visualização dos resultados

Os alunos deverão apresentar visualizações que permitam comparar as diferentes abordagens.

Sempre que possível, deverão apresentar:

1. Os dados originais;
2. Os clusters encontrados pelo K-Means;
3. O grafo de similaridade;
4. As comunidades detectadas;
5. Uma comparação visual entre clustering e comunidades.

---

## 13. Implementação

A implementação deverá ser realizada preferencialmente em **Python**.

Poderão ser utilizadas bibliotecas para:

- Leitura e preparação dos dados;
- Cálculo de distâncias;
- Construção do grafo;
- Visualização;
- Cálculo das métricas.

Entretanto, os alunos deverão implementar **pelo menos um dos algoritmos principais de detecção de comunidades**, em vez de simplesmente utilizar uma implementação pronta.