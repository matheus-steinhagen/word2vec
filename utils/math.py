"""
Meu Laboratório de IA — Word2Vec (Skip-gram + Negative Sampling)
"""

import random
import numpy as np

from utils.text import (
    tokenize,
    vocabulary,
    embedding_generator,
)

# =========================
# Funções matemáticas
# =========================

def sigmoid(x: float) -> float:
    # versão estável
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norm1 * norm2))

def normalize(v):
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm

def sgd_update(v1, v2, alpha, y):
    dot = np.dot(v1, v2)
    sig = sigmoid(dot)
    grad = sig - y

    v1 -= alpha * grad * v2.copy()
    v2 -= alpha * grad * v1 copy()

    orig_v1 = v1.copy()
    orig_v2 = v2.copy()

    v1 -= alpha * grad * orig_v2
    v2 -= alpha * grad * orig_v1

    # evita explosão dos vetores
    normalize(v1)
    normalize(v2)

def similarity(e1, e2):
    return cosine_similarity(e1, e2)

# =========================
# Carregamento
# =========================

def load_corpus_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

corpora = load_corpus_from_txt("./harrypotter.txt")

# =========================
# Configurações
# =========================

embedding_dims = 100
alpha = 0.005
epochs = 50
window = 3
negative_samples = 5

# =========================
# Pré-processamento
# =========================

tokens = tokenize(corpora)
vocab = vocabulary(tokens)

input_embeddings = embedding_generator(vocab, embedding_dims)
output_embeddings = embedding_generator(vocab, embedding_dims)

# =========================
# Monitoramento
# =========================

monitor_pairs = [
    ("harry", "rony"),
    ("harry", "hermione"),
    ("dumbledore", "voldemort"),
]

# =========================
# Treinamento
# =========================

for epoch in range(epochs):

    for i, token in enumerate(tokens):

        central_vector = input_embeddings[token]

        for j in range(-window, window + 1):
            if j == 0:
                continue

            context_index = i + j

            if context_index < 0 or context_index >= len(tokens):
                continue

            context_token = tokens[context_index]

            # POSITIVO
            sgd_update(
                central_vector,
                output_embeddings[context_token],
                alpha,
                y=1
            )

            # NEGATIVOS
            for _ in range(negative_samples):
                negative_token = random.choice(vocab)

                if negative_token == context_token:
                    continue

                sgd_update(
                    central_vector,
                    output_embeddings[negative_token],
                    alpha,
                    y=0
                )

    # =========================
    # FEEDBACK
    # =========================
    print(f"\n[Epoch {epoch}]")

    for w1, w2 in monitor_pairs:
        if w1 in input_embeddings and w2 in input_embeddings:
            sim = similarity(
                input_embeddings[w1],
                input_embeddings[w2]
            )
            print(f"{w1} ~ {w2}: {sim:.4f}")

# =========================
# Inferência
# =========================

while True:
    word = input("Digite uma palavra: ").lower().strip()

    if word not in input_embeddings:
        print("Palavra fora do vocabulário")
        continue

    scores = []

    for other in input_embeddings:
        score = similarity(
            input_embeddings[word],
            input_embeddings[other]
        )
        scores.append((other, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nMais similares:")
    for w, s in scores[:10]:
        print(f"{w} → {s:.4f}")