"""
Módulo de visualização e monitoramento do treino
Responsabilidade: análise, gráficos e métricas
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


class TrainingMonitor:
    def __init__(self):
        self.loss_history = []
        self.norm_history = []

    # =============================
    # LOGS
    # =============================

    def log_epoch(self, loss, embeddings):
        self.loss_history.append(loss)
        self.norm_history.append(self._avg_norm(embeddings))

    def _avg_norm(self, embeddings):
        norms = [np.linalg.norm(v) for v in embeddings.values()]
        return sum(norms) / len(norms)

    # =============================
    # PLOTS
    # =============================

    def plot_loss(self):
        plt.figure()
        plt.title("Loss ao longo das épocas")
        plt.plot(self.loss_history)
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.show()

    def plot_norm(self):
        plt.figure()
        plt.title("Norma média dos embeddings")
        plt.plot(self.norm_history)
        plt.xlabel("Época")
        plt.ylabel("Norma")
        plt.show()

    def plot_embeddings(self, embeddings, words):
        valid_words = [w for w in words if w in embeddings]

        if len(valid_words) < 2:
            print("Poucas palavras para plotar")
            return

        vectors = np.array([embeddings[w] for w in valid_words], dtype=float)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        plt.figure()
        plt.title("Projeção 2D dos embeddings")

        for i, word in enumerate(valid_words):
            x, y = reduced[i]
            plt.scatter(x, y)
            plt.text(x, y, word)

        plt.show()

    # =============================
    # DEBUG DE SIMILARIDADE
    # =============================

    def print_similarity(self, embeddings, dot_product, sigmoid, pairs):
        print("\n=== Similaridades ===")
        for a, b in pairs:
            if a in embeddings and b in embeddings:
                score = sigmoid(dot_product(embeddings[a], embeddings[b]))
                print(f"{a} ~ {b} → {score:.4f}")
        print("=====================\n")