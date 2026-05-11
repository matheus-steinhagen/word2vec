import numpy as np
import random

# ==================================================
# TOKENIZADOR
# Transforma texto em lista de palavras normalizadas
# ==================================================
def tokenize(corpora: str) -> list:
    replacements = {
        ".": "",
        ",": "",
        "-": "",
        ":": "",
        "?": "",
        "!": ""
    }

    for k, v in replacements.items():
        corpora = corpora.replace(k, v)

    return corpora.lower().strip().split()

# ==================================================
# VOCABULARIO
# Cria uma lista de palavras únicas do corpus
# ==================================================
def vocabulary(tokens: list) -> list:
    return list(set(tokens))

# ==================================================
# GERADOR DE EMBEDDINGS
# ==================================================
def embedding_generator(vocabulary_list: list[str], embeddings_dims: int) -> dict:
    return  {word: np.random.uniform(-0.01, 0.01, embeddings_dims) for word in vocabulary_list}

# ==================================================
# CÓPIA PROFUNDA
# ==================================================
def deep_copy(embeddings):
    return {word: emb.copy() for word, emb in embeddings.items()}