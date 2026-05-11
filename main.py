"""
Word2Vec — Skip-gram + Negative Sampling (SGNS)
Corpus em portugues, encoding detectado automaticamente.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. PRE-PROCESSAMENTO
# ===========================================================================

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


def tokenize(text: str) -> list[str]:
    """
    Tokenizacao Unicode para portugues.

    [^\W\d_] = qualquer letra Unicode, incluindo acentos e cedilha.
    Antes: 'capitulo' -> ['capit', 'culo']   (regex [a-z]+ quebra em acentos)
    Agora: 'capitulo' -> ['capitulo']
    """
    text = text.lower()
    return re.findall(r"[^\W\d_]+", text, re.UNICODE)


def build_vocabulary(
    tokens: list[str],
    min_count: int = 5,
) -> tuple[list[str], dict[str, int], Counter]:
    counts = Counter(tokens)
    vocab = [w for w, c in counts.most_common() if c >= min_count]
    token2idx = {w: i for i, w in enumerate(vocab)}
    log.info(
        "Vocabulario: %d tokens unicos (min_count=%d, corpus=%d tokens)",
        len(vocab), min_count, len(tokens),
    )
    return vocab, token2idx, counts


def subsample_tokens(
    tokens: list[str],
    counts: Counter,
    total: int,
    t: float = 1e-4,
) -> list[str]:
    """Descarta ocorrencias de palavras muito frequentes — Mikolov eq. (5)."""
    result: list[str] = []
    for token in tokens:
        freq = counts[token] / total
        if freq == 0:
            continue
        keep_prob = min(1.0, (np.sqrt(freq / t) + 1) * (t / freq))
        if random.random() < keep_prob:
            result.append(token)
    log.info(
        "Subsampling: %d -> %d tokens (%.1f%% mantidos)",
        total, len(result), 100.0 * len(result) / total,
    )
    return result


def build_noise_distribution(
    vocab: list[str], counts: Counter, power: float = 0.75
) -> np.ndarray:
    """P(w) proporcional a freq(w)^0.75."""
    freqs = np.array([counts[w] ** power for w in vocab], dtype=np.float64)
    freqs /= freqs.sum()
    return freqs


# ===========================================================================
# 2. MATEMATICA
# ===========================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    return 0.0 if n1 == 0 or n2 == 0 else float(np.dot(v1, v2) / (n1 * n2))


# ===========================================================================
# 3. MODELO
# ===========================================================================

class Word2Vec:
    """
    Skip-gram com Negative Sampling.

    W_in  — vetores da palavra central (usados na inferencia)
    W_out — vetores de contexto        (descartados apos treino)

    Hiperparametros recomendados para ~1 livro PT-BR:
        dim=100, window=5, neg_samples=5, alpha=0.025, epochs=5
    """

    def __init__(
        self,
        vocab: list[str],
        token2idx: dict[str, int],
        counts: Counter,
        dim: int = 100,
        window: int = 5,
        neg_samples: int = 5,
        alpha: float = 0.025,
        min_alpha: float = 0.0001,
    ) -> None:
        self.vocab = vocab
        self.token2idx = token2idx
        self.counts = counts
        self.vocab_size = len(vocab)
        self.dim = dim
        self.window = window
        self.neg_samples = neg_samples
        self.alpha_0 = alpha
        self.min_alpha = min_alpha

        scale = 0.5 / dim
        self.W_in  = np.random.uniform(-scale, scale, (self.vocab_size, dim))
        self.W_out = np.zeros((self.vocab_size, dim))

        self.noise_dist = build_noise_distribution(vocab, counts)
        self._vocab_indices = np.arange(self.vocab_size)

        log.info(
            "Modelo: vocab=%d  dim=%d  window=%d  neg=%d  alpha=%.4f",
            self.vocab_size, dim, window, neg_samples, alpha,
        )

    def _current_alpha(self, progress: float) -> float:
        return max(self.min_alpha, self.alpha_0 * (1.0 - progress) + self.min_alpha * progress)

    def _sgd_update(
        self,
        center_idx: int,
        context_idx: int,
        neg_idxs: np.ndarray,
        alpha: float,
    ) -> float:
        """
        Loss: L = -log sigmoid(v_c . v_o) - sum log sigmoid(-v_c . v_n)

        Snapshot do centro salvo antes de modificacoes.
        SEM normalizacao dos vetores durante treino.
        """
        center_vec = self.W_in[center_idx].copy()

        ctx_vec = self.W_out[context_idx]
        dot_pos = float(np.dot(center_vec, ctx_vec))
        sig_pos = float(sigmoid(np.array([dot_pos]))[0])
        err_pos = sig_pos - 1.0

        neg_vecs = self.W_out[neg_idxs]
        dots_neg = neg_vecs @ center_vec
        errs_neg = sigmoid(dots_neg)

        grad_center = err_pos * ctx_vec + errs_neg @ neg_vecs

        self.W_out[context_idx] -= alpha * err_pos * center_vec
        self.W_out[neg_idxs]    -= alpha * errs_neg[:, None] * center_vec[None, :]
        self.W_in[center_idx]   -= alpha * grad_center

        loss = -np.log(sig_pos + 1e-9) - np.sum(np.log(sigmoid(-dots_neg) + 1e-9))
        return float(loss)

    def _sample_negatives(self, exclude_idx: int) -> np.ndarray:
        candidates = np.random.choice(
            self._vocab_indices, size=self.neg_samples * 2, p=self.noise_dist
        )
        candidates = candidates[candidates != exclude_idx]
        return candidates[: self.neg_samples]

    def train(
        self,
        tokens: list[str],
        epochs: int = 5,
        monitor_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        indexed = [self.token2idx[t] for t in tokens if t in self.token2idx]
        n = len(indexed)
        total_steps = epochs * n
        step = 0

        log.info("Treino: %d epocas x %d tokens = %d passos", epochs, n, total_steps)
        t_start = time.time()

        for epoch in range(epochs):
            epoch_loss, epoch_updates = 0.0, 0
            t_epoch = time.time()

            for i, center_idx in enumerate(indexed):
                alpha = self._current_alpha(step / total_steps)
                step += 1
                win = random.randint(1, self.window)

                for j in range(-win, win + 1):
                    if j == 0:
                        continue
                    ci = i + j
                    if ci < 0 or ci >= n:
                        continue
                    context_idx = indexed[ci]
                    neg_idxs = self._sample_negatives(context_idx)
                    epoch_loss += self._sgd_update(center_idx, context_idx, neg_idxs, alpha)
                    epoch_updates += 1

            log.info(
                "Epoch %d/%d | loss=%.4f | alpha=%.6f | %.1fs",
                epoch + 1, epochs,
                epoch_loss / max(epoch_updates, 1),
                self._current_alpha((epoch + 1) / epochs),
                time.time() - t_epoch,
            )
            if monitor_pairs:
                self._report_similarities(monitor_pairs)

        log.info("Treino concluido em %.1fs", time.time() - t_start)

    def _report_similarities(self, pairs: list[tuple[str, str]]) -> None:
        for w1, w2 in pairs:
            if w1 in self.token2idx and w2 in self.token2idx:
                sim = cosine_similarity(
                    self.W_in[self.token2idx[w1]],
                    self.W_in[self.token2idx[w2]],
                )
                log.info("  sim(%-16s, %-16s) = %.4f", w1, w2, sim)
            else:
                missing = [w for w in (w1, w2) if w not in self.token2idx]
                log.warning("  par ignorado — fora do vocab: %s", missing)

    def _normed_matrix(self) -> np.ndarray:
        norms = np.linalg.norm(self.W_in, axis=1, keepdims=True)
        return self.W_in / np.where(norms == 0, 1.0, norms)

    def most_similar(self, word: str, k: int = 10) -> list[tuple[str, float]]:
        if word not in self.token2idx:
            raise KeyError(f"'{word}' nao esta no vocabulario.")
        idx = self.token2idx[word]
        normed = self._normed_matrix()
        sims = normed @ normed[idx]
        sims[idx] = -np.inf
        top_k = np.argpartition(sims, -k)[-k:]
        top_k = top_k[np.argsort(sims[top_k])[::-1]]
        return [(self.vocab[i], float(sims[i])) for i in top_k]

    def analogy(self, pos1: str, neg1: str, pos2: str, k: int = 5) -> list[tuple[str, float]]:
        """Analogia vetorial: pos1 - neg1 + pos2 = ?"""
        for w in (pos1, neg1, pos2):
            if w not in self.token2idx:
                raise KeyError(f"'{w}' nao esta no vocabulario.")
        v = (
            self.W_in[self.token2idx[pos1]]
            - self.W_in[self.token2idx[neg1]]
            + self.W_in[self.token2idx[pos2]]
        )
        v = v / (np.linalg.norm(v) + 1e-9)
        normed = self._normed_matrix()
        sims = normed @ v
        for w in (pos1, neg1, pos2):
            sims[self.token2idx[w]] = -np.inf
        top_k = np.argpartition(sims, -k)[-k:]
        top_k = top_k[np.argsort(sims[top_k])[::-1]]
        return [(self.vocab[i], float(sims[i])) for i in top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"dim": self.dim, "vocab": self.vocab, "embeddings": self.W_in.tolist()},
                f, ensure_ascii=False,
            )
        log.info("Salvo em %s (%.1f MB)", path, path.stat().st_size / 1e6)

    @classmethod
    def load(cls, path: str | Path) -> "Word2Vec":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = data["vocab"]
        token2idx = {w: i for i, w in enumerate(vocab)}
        model = cls(
            vocab=vocab, token2idx=token2idx,
            counts=Counter({w: 1 for w in vocab}),
            dim=data["dim"],
        )
        model.W_in = np.array(data["embeddings"], dtype=np.float64)
        log.info("Carregado: %d palavras, dim=%d", len(vocab), data["dim"])
        return model


# ===========================================================================
# 4. MAIN
# ===========================================================================

def main() -> None:
    CORPUS_PATH = "./harrypotter.txt"
    SAVE_PATH   = "./embeddings.json"

    DIM         = 50    # 300 so vale com >10M tokens; aqui causa overfitting
    WINDOW      = 5
    NEG_SAMPLES = 5      # acima de 10 prejudica; 5 e o padrao do paper
    ALPHA       = 0.02  # padrao Mikolov; 0.005 causa colapso de embeddings
    MIN_ALPHA   = 0.0001
    EPOCHS      = 20
    MIN_COUNT   = 5
    T_SUBSAMPLE = 1e-4

    MONITOR_PAIRS = [
        ("harry",      "hermione"),
        ("harry",      "rony"),
        ("dumbledore", "voldemort"),
        ("hogwarts",   "escola"),
        ("correu",   "chuva"),
    ]

    STOPWORDS = {
        "o","a","os","as","um","uma","uns","umas",
        "de","do","da","dos","das","em","no","na","nos","nas",
        "por","com","para","sem","sob","sobre", "que", "só", "nele", "nela",
        "e","ou","mas","nem","pois","porém","logo","portanto",

        "isso", "ah", "demais", "nunca", "sim", "não", "ora", "já", "agora", "aqui", "ali", "isso", "esse", "essa"
    }

    corpus = load_corpus(CORPUS_PATH)
    tokens_raw = tokenize(corpus)

    # 1. REMOVE STOPWORDS PRIMEIRO
    tokens_clean = [t for t in tokens_raw if t not in STOPWORDS]

    # 2. BUILD VOCAB JÁ LIMPO
    vocab, token2idx, counts = build_vocabulary(tokens_clean, min_count=MIN_COUNT)

    # 3. FILTRA OOV
    tokens_filtered = [t for t in tokens_clean if t in token2idx]

    # 4. SUBSAMPLING
    tokens = subsample_tokens(tokens_filtered, counts, len(tokens_filtered), t=T_SUBSAMPLE)

    model = Word2Vec(
        vocab, token2idx, counts,
        dim=DIM, window=WINDOW, neg_samples=NEG_SAMPLES,
        alpha=ALPHA, min_alpha=MIN_ALPHA,
    )
    model.train(tokens, epochs=EPOCHS, monitor_pairs=MONITOR_PAIRS)
    model.save(SAVE_PATH)

    print("\n" + "=" * 60)
    print("  Word2Vec PT-BR — modo inferencia")
    print("  <palavra>                    vizinhos mais proximos")
    print("  :analogia <p1> <n1> <p2>    analogia vetorial")
    print("  :carregar / :sair")
    print("=" * 60)

    while True:
        try:
            cmd = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not cmd or cmd == ":sair":
            break

        if cmd == ":carregar":
            model = Word2Vec.load(SAVE_PATH)
            continue

        if cmd.startswith(":analogia"):
            parts = cmd.split()
            if len(parts) != 4:
                print("Uso: :analogia pos1 neg1 pos2")
                continue
            _, p1, n1, p2 = parts
            try:
                print(f"\n{p1} - {n1} + {p2} =~")
                for w, s in model.analogy(p1, n1, p2):
                    print(f"  {w:<24} {s:.4f}")
            except KeyError as e:
                print(e)
            continue

        try:
            results = model.most_similar(cmd, k=10)
            print(f"\nMais similares a '{cmd}':")
            for rank, (w, s) in enumerate(results, 1):
                bar = "█" * int(max(s, 0) * 30)
                print(f"  {rank:2}. {w:<24} {s:.4f}  {bar}")
        except KeyError:
            sugestoes = [v for v in model.vocab if cmd in v or v in cmd][:5]
            print(f"  '{cmd}' nao esta no vocabulario.")
            if sugestoes:
                print(f"  Palavras parecidas: {', '.join(sugestoes)}")


if __name__ == "__main__":
    main()