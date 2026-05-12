from __future__ import annotations

import logging
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

# === LOGGING CONFIG ==========

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# === DETECT ENCODING ==========

def detect_encoding(path: str | Path) -> str:
    """
    Detecta encoding via chardet.
    Fallback: latin-1 (nunca falha, cobre Windows-1252 e ISO-8859).
    """
    path = Path(path)
    raw = path.read_bytes()
    try:
        import chardet

        result = chardet.detect(raw[:100_000])
        enc = result.get("encoding") or "latin-1"
        conf = result.get("confidence", 0)

        log.info("Encoding detectado: %s (confianca=%.0f%%)", enc, conf * 100)
        return enc
    
    except ImportError:
        log.warning("chardet nao instalado — tentando UTF-8 depois latin-1")

        try:
            raw.decode("utf-8")
            return "utf-8"

        except UnicodeDecodeError:
            return "latin-1"

# === LOAD CORPUS ==========

def load_corpus(path: str | Path) -> str:
    path = Path(path)
    enc = detect_encoding(path)

    try:
        text = path.read_text(encoding=enc)

    except (UnicodeDecodeError, LookupError):
        log.warning("Encoding '%s' falhou — usando latin-1", enc)
        text = path.read_text(encoding="latin-1")

    log.info("Corpus carregado: %d caracteres", len(text))
    return text

# === TOKENIZE ==========

def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"[^\W\d_]+", text, re.UNICODE)

# === BUILD VOCABULARY ==========

def build_vocabulary(
    tokens: list[str],
    min_count: int = 5,
) -> tuple[list[str], dict[str, int], Counter]:

    # Conta quantas vezes cada token aparece no corpus
    counts = Counter(tokens)

    # Cria o vocabulário ordenado por frequência decrescente
    # e remove tokens com frequência menor que min_count
    vocab = [w for w, c in counts.most_common() if c >= min_count]

    # Cria um mapeamento token -> índice
    # Exemplo: {"python": 0, "dados": 1}
    token2idx = {w: i for i, w in enumerate(vocab)}

    # Exibe informações sobre o vocabulário gerado
    log.info(
        "Vocabulario: %d tokens unicos (min_count=%d, corpus=%d tokens)",
        len(vocab), min_count, len(tokens),
    )

    # Retorna:
    # - lista de tokens do vocabulário
    # - mapeamento token -> índice
    # - contagem de frequência dos tokens
    return vocab, token2idx, counts

# === SUBSAMPLE TOKENS ==========

def subsample_tokens(
    tokens: list[str],
    counts: Counter,
    total: int,
    t: float = 1e-4,
) -> list[str]:
    """
    Remove aleatóriamente ocorrências de palavras muito frequentes
    
    Objetivo:
    Reduzir o ruído e acelerar o treinamento

    Implementa o subsampling do paper Word2Vec (Mikolov et al.).
    """

    # Inicia uma lista vazia de strings
    result: list[str] = []

    # Iteração nos tokens
    for token in tokens:

        # Frequencia relativa do token no corpus
        freq = counts[token] / total

        # Proteção de divisão por zero
        if freq == 0:
            continue

        # Probabilidade de manter o token:
        # palavras muito frequentes têm menor chance de permanecer
        keep_prob = min(1.0, (np.sqrt(freq / t) + 1) * (t / freq))

        # Mantém o token aleatoriamente baseado na probabilidade calculada
        if random.random() < keep_prob:
            result.append(token)

    log.info(
        "Subsampling: %d -> %d tokens (%.1f%% mantidos)",
        total, len(result), 100.0 * len(result) / total,
    )
    return result

# === BUILD NOISE DISTRIBUTION ==========

def build_noise_distribution(
    vocab: list[str],
    counts: Counter,
    power: float = 0.75
) -> np.ndarray:
    """
    Cria a distribuição de probabilidade usada no negative Sampling
    
    Cada palavra recebe probabilidade proporcional a:
        freq(w)^0.75

    Isso reduz o domínio excessivo de palavras muito frequentes
    e melhora a qualidade dos exemplos negativos.
    """

    # Eleva a frequencia do token poe power
    # Isso ajuda a diminuir a distância entre palavras de altíssima e baixíssima frequencia
    freqs = np.array([counts[w] ** power for w in vocab], dtype=np.float64)

    # Normalização
    # transforma em distribuição de probabilidade
    freqs /= freqs.sum()
    
    return freqs