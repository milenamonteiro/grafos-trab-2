"""Projeto 2: Clustering de dados e detecção de comunidades em grafos.

Este script implementa, de forma organizada, o fluxo completo do projeto:
1. carregamento e padronização dos dados;
2. escolha do número de clusters via silhouette score;
3. k-means;
4. construção de grafo k-NN com pesos de similaridade;
5. detecção de comunidades via Louvain;
6. comparação entre clustering e comunidades usando Adjusted Rand Index;
7. visualizações relevantes.

Observação: o código foi escrito para organização e clareza, sem execução neste momento.
"""

import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# 1. Carregamento e preparação dos dados
# -----------------------------------------------------------------------------
def carregar_dados_iris():
    """Carrega o dataset Iris e retorna DataFrame com atributos e alvo."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    # display random 5 rows of X dataframe w display
    display(X.sample(5))
    
    
    y = pd.Series(iris.target, name="classe")
    return X, y

"""padronizar y usando a formula
$$
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

onde:
 $\mu_j$ é a média do atributo $j$;
$\sigma_j$ é o desvio-padrão do atributo $j$.
"""


def padronizar_dados(X):
    """Padroniza as colunas numéricas com z-score."""
    scaler = StandardScaler()
    X_pad = scaler.fit_transform(X)
    X_pad = pd.DataFrame(X_pad, columns=X.columns)
    return X_pad, scaler


# -----------------------------------------------------------------------------
# 2. Escolha do número de clusters
# -----------------------------------------------------------------------------
def escolher_k_por_silhouette(X, k_min=2, k_max=10):
    """Escolhe k com base no maior valor de silhouette score."""
    melhores_scores = []

    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = modelo.fit_predict(X)

        if k == 1:
            continue

        score = silhouette_score(X, labels)
        melhores_scores.append((k, score))

    if not melhores_scores:
        raise ValueError("Não foi possível calcular silhouette score para os valores de k informados.")

    melhor_k, melhor_score = max(melhores_scores, key=lambda item: item[1])
    return melhor_k, melhor_score, melhores_scores


def escolher_k_por_inercia(X, k_min=2, k_max=10):
    """Retorna a curva de inércia para auxiliar na visualização do elbow method."""
    inercia = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X)
        inercia.append((k, modelo.inertia_))
    return inercia


# -----------------------------------------------------------------------------
# 3. Clustering tradicional: K-Means
# -----------------------------------------------------------------------------
def executar_kmeans(X, k):
    """Executa KMeans e retorna rótulos e modelo."""
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X)
    return modelo, labels


# -----------------------------------------------------------------------------
# 4. Grafo de similaridade (k-NN)
# -----------------------------------------------------------------------------
def construir_grafo_knn(X, k=10, sigma=1.0):
    """Constrói um grafo k-NN ponderado pela similaridade gaussiana.

    Cada nó representa um objeto. A aresta entre i e j existe se j estiver entre
    os k vizinhos mais próximos de i (ou vice-versa, em grafos não direcionados).
    """
    X_np = X.to_numpy()
    dist = euclidean_distances(X_np, X_np)
    np.fill_diagonal(dist, np.inf)

    G = nx.Graph()
    G.add_nodes_from(range(len(X_np)))

    for i in range(len(X_np)):
        vizinhos = np.argsort(dist[i])[:k]
        for j in vizinhos:
            if i == j:
                continue

            d_ij = dist[i, j]
            peso = np.exp(-(d_ij ** 2) / (2 * sigma ** 2))
            G.add_edge(i, j, weight=float(peso))

    return G


# -----------------------------------------------------------------------------
# 5. Detecção de comunidades
# -----------------------------------------------------------------------------
def detectar_comunidades_louvain(G):
    """Aplica Louvain e retorna lista de comunidades e rótulos por nó."""
    comunidades = louvain_communities(G, weight="weight", seed=42)

    rotulos = np.full(G.number_of_nodes(), -1, dtype=int)
    for idx, comunidade in enumerate(comunidades):
        for no in comunidade:
            rotulos[no] = idx

    return comunidades, rotulos


# -----------------------------------------------------------------------------
# 6. Comparação entre partições
# -----------------------------------------------------------------------------
def comparar_particoes(labels_cluster, labels_comunidade):
    """Calcula o Adjusted Rand Index entre duas partições."""
    ari = adjusted_rand_score(labels_cluster, labels_comunidade)
    return ari


# -----------------------------------------------------------------------------
# 7. Visualizações
# -----------------------------------------------------------------------------
def plotar_curva_elbow(inercia):
    """Plota a curva de inércia para o elbow method."""
    ks = [k for k, _ in inercia]
    valores = [valor for _, valor in inercia]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, valores, marker="o")
    plt.title("Método do cotovelo (Elbow Method)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return plt.gcf()


def plotar_silhouette_scores(scores):
    """Plota os valores de silhouette score para diferentes k."""
    ks = [k for k, _ in scores]
    valores = [score for _, score in scores]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, valores, marker="o", color="green")
    plt.title("Silhouette Score por número de clusters")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return plt.gcf()


def plotar_grafo(G):
    """Visualiza a estrutura do grafo k-NN."""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 8))
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="skyblue",
        node_size=80,
        edge_color="gray",
        width=1.2,
    )
    plt.title("Grafo k-NN de similaridade")
    plt.tight_layout()
    return plt.gcf()


# -----------------------------------------------------------------------------
# 8. Pipeline principal
# -----------------------------------------------------------------------------
def pipeline_projeto_2():
    """Executa o pipeline completo do projeto e imprime os principais resultados."""
    X, y = carregar_dados_iris()
    X_pad, _ = padronizar_dados(X)

    # Escolha do número de clusters
    melhor_k, melhor_score, silhouette_scores = escolher_k_por_silhouette(X_pad)
    inercia = escolher_k_por_inercia(X_pad)

    print("\n=== Escolha do número de clusters ===")
    print(f"Melhor k pelo silhouette score: {melhor_k} (score = {melhor_score:.4f})")
    print("Silhouette scores por k:")
    for k, score in silhouette_scores:
        print(f"  k = {k:2d} -> silhouette = {score:.4f}")

    # Clustering tradicional
    modelo, labels_cluster = executar_kmeans(X_pad, melhor_k)
    print("\n=== K-Means ===")
    print(f"Centroides: {modelo.cluster_centers_}")
    print(f"Rótulos do cluster: {labels_cluster[:10]}")

    # Construção do grafo de similaridade
    G = construir_grafo_knn(X_pad, k=10, sigma=1.0)
    comunidades, labels_comunidade = detectar_comunidades_louvain(G)

    print("\n=== Comunidades ===")
    print(f"Número de comunidades encontradas: {len(comunidades)}")
    for i, comunidade in enumerate(comunidades):
        print(f"  Comunidade {i}: {len(comunidade)} nós")

    # Comparação
    ari = comparar_particoes(labels_cluster, labels_comunidade)
    print(f"\nAdjusted Rand Index (cluster vs comunidades): {ari:.4f}")

    # Modularidade
    modularidade = nx.algorithms.community.quality.modularity(G, comunidades, weight="weight")
    print(f"Modularidade: {modularidade:.4f}")

    # Visualizações
    plotar_curva_elbow(inercia)
    plotar_silhouette_scores(silhouette_scores)
    plotar_grafo(G)

    plt.show()

    return {
        "X": X_pad,
        "labels_cluster": labels_cluster,
        "labels_comunidade": labels_comunidade,
        "comunidades": comunidades,
        "ARI": ari,
        "modularidade": modularidade,
        "k": melhor_k,
    }


if __name__ == "__main__":
    pipeline_projeto_2()
