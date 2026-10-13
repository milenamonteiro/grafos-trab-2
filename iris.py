# Anna Beatriz Souza e Silva - 13694103
# Milena Rodrigues Monteiro - 12566157


# ==========================================
# IMPORTANDO AS BIBLIOTECAS
# ==========================================
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community.quality import modularity
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# Cria o diretório para salvar as fotos, se não existir
os.makedirs('fotos', exist_ok=True)


# ==========================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==========================================
iris = load_iris()
X = iris.data
y_real = iris.target
feature_names = iris.feature_names
target_names = list(iris.target_names)

df = pd.DataFrame(X, columns=feature_names)
df['especie_real'] = [target_names[i] for i in y_real]
print(df.head(), "\n")

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redução de dimensionalidade para visualização 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ==========================================
# 2. VISUALIZAÇÃO DOS DADOS ORIGINAIS
# ==========================================
# Sem rótulos
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], color='blue', edgecolor='k', s=60)
plt.title('Dados Originais (Iris - Sem Rótulos)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True, alpha=0.3)
plt.savefig('fotos/dados_originais.png', dpi=150, bbox_inches='tight')
plt.close()

# Com rótulos reais
plt.figure(figsize=(7, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_real, cmap='viridis', edgecolor='k', s=60)
plt.title('Dados Originais (Iris - Com Rótulos)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(handles=scatter.legend_elements()[0], labels=target_names)
plt.grid(True, alpha=0.3)
plt.savefig('fotos/dados_originais_com_rotulo.png', dpi=150, bbox_inches='tight')
plt.close()


# ==========================================
# 3. ALGORITMO K-MEANS
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
ari_kmeans_real = adjusted_rand_score(y_kmeans, y_real)

plt.figure(figsize=(7, 5))
scatter_km = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='Set1', edgecolor='k', s=60)
plt.title('Clusters Obtidos pelo K-Means (K=3)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(handles=scatter_km.legend_elements()[0], labels=[f'Cluster {i}' for i in range(3)])
plt.grid(True, alpha=0.3)
plt.savefig('fotos/clusters_kmeans.png', dpi=150, bbox_inches='tight')
plt.close()


# ==========================================
# 4. CONSTRUÇÃO DO GRAFO DE SIMILARIDADE (KNN)
# ==========================================
k_base = 3
matriz_adj = kneighbors_graph(X_scaled, n_neighbors=k_base, mode='connectivity', include_self=False)
G_base = nx.from_scipy_sparse_array(matriz_adj)

pos = nx.spring_layout(G_base, seed=42)

plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G_base, pos, node_size=60, node_color='lightblue', edgecolors='k')
nx.draw_networkx_edges(G_base, pos, alpha=0.3)
plt.title(f'Grafo de Similaridade k-NN (k={k_base})')
plt.axis('off')
plt.savefig('fotos/grafo_base.png', dpi=150, bbox_inches='tight')
plt.close()


# ==========================================
# 5. ALGORITMO LABEL PROPAGATION
# ==========================================
def label_propagation(G, max_iter=100, seed=42):
    rng = random.Random(seed)
    labels = {node: node for node in G.nodes()}
    nodes = list(G.nodes())
    
    for iteracao in range(max_iter):
        rng.shuffle(nodes)
        mudou = False
        
        for node in nodes:
            vizinhos = list(G.neighbors(node))
            if not vizinhos:
                continue
            
            rotulos_vizinhos = [labels[v] for v in vizinhos]
            contagem = Counter(rotulos_vizinhos)
            max_votos = max(contagem.values())
            
            candidatos = [rot for rot, freq in contagem.items() if freq == max_votos]
            novo_rotulo = rng.choice(candidatos)
            
            if labels[node] != novo_rotulo:
                labels[node] = novo_rotulo
                mudou = True
                
        if not mudou:
            break
            
    return labels

# ==========================================
# 6. ANÁLISE VARIANDO O PARÂMETRO K
# ==========================================
k_variacoes = [3, 5, 10, 15, 20]
modularidades = {}
ari_lpa_kmeans = {}
ari_lpa_real = {}
num_comunidades_k = {}

for k_var in k_variacoes:
    matriz_adj_var = kneighbors_graph(X_scaled, n_neighbors=k_var, mode='connectivity', include_self=False)
    G_var = nx.from_scipy_sparse_array(matriz_adj_var)

    labels_dict_var = label_propagation(G_var, seed=42)
    
    # Organizar os nós em uma lista de grupos para a modularidade
    comunidades_dict = {}
    for node, comm in labels_dict_var.items():
        if comm not in comunidades_dict:
            comunidades_dict[comm] = set()
        comunidades_dict[comm].add(node)
    comunidades = list(comunidades_dict.values())
    
    if len(comunidades) > 1 and G_var.number_of_edges() > 0:
        mod_q = nx.community.modularity(G_var, comunidades)
    else:
        mod_q = 0.0
    modularidades[k_var] = mod_q

    # Formatar rótulos brutos para calcular o ARI
    rotulos_brutos_var = [labels_dict_var[i] for i in range(len(X_scaled))]
    mapa_rotulos_var = {rot: idx for idx, rot in enumerate(np.unique(rotulos_brutos_var))}
    lpa_labels_var = np.array([mapa_rotulos_var[rot] for rot in rotulos_brutos_var])

    num_comunidades_k[k_var] = len(np.unique(lpa_labels_var))
    ari_lpa_kmeans[k_var] = adjusted_rand_score(lpa_labels_var, y_kmeans)
    ari_lpa_real[k_var] = adjusted_rand_score(lpa_labels_var, y_real)

    # Plot e salvamento
    pos_rede = nx.spring_layout(G_var, seed=42)
    
    plt.figure(figsize=(8, 6))
    plt.title(f'Comunidades Detectadas usando Label Propagation (k={k_var})')
    nx.draw_networkx_edges(G_var, pos=pos_rede, alpha=0.3, edge_color='gray')
    nx.draw_networkx_nodes(G_var, pos=pos_rede, node_color=lpa_labels_var, cmap='Set1', node_size=60, edgecolors='k')
    plt.axis('off') 
    plt.savefig(f'fotos/grafo_k_{k_var}.png', dpi=150, bbox_inches='tight')
    plt.close() 

    print(f"Grafo (k={k_var}) --> Clusters: {len(np.unique(lpa_labels_var))} | Tamanho das comunidades: {dict(Counter(lpa_labels_var))}")

# Gráfico comparativo de métricas
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(list(modularidades.keys()), list(modularidades.values()), marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=7, label='Modularidade Q')
ax1.plot(list(ari_lpa_kmeans.keys()), list(ari_lpa_kmeans.values()), marker='s', linestyle='--', color='tab:red', linewidth=2, markersize=7, label='ARI (LPA x K-Means)')
ax1.plot(list(ari_lpa_real.keys()), list(ari_lpa_real.values()), marker='^', linestyle=':', color='tab:green', linewidth=2, markersize=7, label='ARI (LPA x Ground Truth)')
ax1.plot(k_variacoes, [ari_kmeans_real]*len(k_variacoes), marker='X', linestyle='-', color='tab:purple', linewidth=2, markersize=7, label='ARI (K-Means x Ground Truth)')

ax1.set_title('Impacto do Parâmetro k na Estrutura do Grafo e Comunidades')
ax1.set_xlabel('Parâmetro k (Vizinhos)')
ax1.set_ylabel('Score (Modularidade e ARI)')
ax1.set_xticks(k_variacoes)
ax1.grid(True, linestyle='-', alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(list(num_comunidades_k.keys()), list(num_comunidades_k.values()), marker='D', linestyle='-.', color='black', linewidth=2, markersize=7, label='Nº de Comunidades (LPA)')
ax2.set_ylabel('Quantidade de Comunidades', color='black')

linhas_1, labels_1 = ax1.get_legend_handles_labels()
linhas_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(linhas_1 + linhas_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

plt.subplots_adjust(bottom=0.25)
plt.savefig('fotos/grafico_metricas_k.png', dpi=150, bbox_inches='tight')


# ==========================================
# 7. RESULTADOS FINAIS PARA O GRAFO BASE (k=3) E AVALIAÇÃO
# ==========================================
labels_dict = label_propagation(G_base, seed=42)
rotulos_brutos = [labels_dict[i] for i in range(len(X_scaled))]
mapa_rotulos = {rot: idx for idx, rot in enumerate(np.unique(rotulos_brutos))}
y_lpa = np.array([mapa_rotulos[rot] for rot in rotulos_brutos])

# Visualização final k=3
plt.figure(figsize=(8, 6))
nx.draw_networkx_nodes(G_base, pos, node_size=70, node_color=y_lpa, cmap='Set1', edgecolors='k')
nx.draw_networkx_edges(G_base, pos, alpha=0.3)
plt.title(f'Comunidades Detectadas usando Label Propagation (k={k_base})')
plt.axis('off')
plt.savefig('fotos/grafo_comunidades.png', dpi=150, bbox_inches='tight')
plt.close()

# --- RESULTADOS FINAIS ---

print("\n" + "="*50)
print(f"RESULTADOS FINAIS (Para k={k_base})")
print("="*50)
print(f"Número de comunidades encontradas: {len(np.unique(y_lpa))}")
print(f"Tamanho de cada comunidade: {dict(Counter(y_lpa))}")

# Calcular a modularidade
comunidades_base_dict = {}
for i, rotulo in enumerate(y_lpa):
    if rotulo not in comunidades_base_dict:
        comunidades_base_dict[rotulo] = set()
    comunidades_base_dict[rotulo].add(i)
comunidades_base = list(comunidades_base_dict.values())
mod_base = nx.community.modularity(G_base, comunidades_base)
print(f"\nModularidade (Label Propagation k={k_base}): {mod_base:.4f}")

# Comparar LPA vs K-Means
ari_lpa_km = adjusted_rand_score(y_kmeans, y_lpa)
print(f"ARI (K-Means x Label Propagation): {ari_lpa_km:.4f}")

# Comparar os dois modelos com os labels reais (Ground Truth)
ari_lpa_real_final = adjusted_rand_score(y_real, y_lpa)
print(f"\nARI (K-Means x Real): {ari_kmeans_real:.4f}")
print(f"ARI (Label Propagation x Real): {ari_lpa_real_final:.4f}")