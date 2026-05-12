from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pre_processing import build_noise_distribution
from utils.math import sigmoid, cosine_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

class Word2Vec:
    """
    Skip-gram com Negative Sampling.

    W_in  — vetores da palavra central (usados na inferencia)
    W_out — vetores de contexto        (descartados apos treino)
    """

    # === INICIALIZAR ==========

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

        # Recepção de parâmetros
        self.vocab = vocab
        self.token2idx = token2idx
        self.counts = counts
        self.vocab_size = len(vocab)
        self.dim = dim
        self.window = window
        self.neg_samples = neg_samples
        self.alpha_0 = alpha
        self.min_alpha = min_alpha

        # Matrizes de entrada e saída
        scale = 0.5 / dim
        self.W_in  = np.random.uniform(-scale, scale, (self.vocab_size, dim))
        self.W_out = np.zeros((self.vocab_size, dim))

        # Distribuição de probabilidade
        self.noise_dist = build_noise_distribution(vocab, counts)

        # Cria um array com os índices de todas as palavras no vocabulário
        self._vocab_indices = np.arange(self.vocab_size)

        log.info(
            "Modelo: vocab=%d  dim=%d  window=%d  neg=%d  alpha=%.4f",
            self.vocab_size, dim, window, neg_samples, alpha,
        )

    # === HELPERS INTERNOS ==========

    # Obter alpha atual
    def _current_alpha(self, progress: float) -> float:
        """
        Aprendizado está longe no início → learning_rate alta,
        no final, aprendizado está próximo → learning_rate baixa para refinamento
        """
        # Garante que o valor final nunca seja menor que self.min_alpha.
        return max(self.min_alpha, self.alpha_0 * (1.0 - progress) + self.min_alpha * progress)

    # Atualizar distância sigmoid
    def _sgd_update(
        self,
        center_idx: int,        # índice da palavra central
        context_idx: int,       # índice da palavra de contexto positiva
        neg_idxs: np.ndarray,   # índices das palavras negativas (negative samples)
        alpha: float,           # learning_rate
    ) -> float:
        """
        Atualiza SGD do Skip-Gram com Negative Sampling

        Ou seja:
        - calcula similaridades
        - mede erro
        - calcula gradientes
        - ajusta embeddings
        - retorna a loss

        Loss: L = -log sigmoid(v_c . v_o) - sum log sigmoid(-v_c . v_n)
        """

        center_vec = self.W_in[center_idx].copy()       # vetor da palavra central

        ctx_vec = self.W_out[context_idx]               # vetor de saída da palavra de contexto
        dot_pos = float(np.dot(center_vec, ctx_vec))    # Similaridade linear entre centro e contexto positivo
        sig_pos = float(sigmoid(np.array([dot_pos]))[0])# Sigmoide → probabilidade prevista do par positivo
        err_pos = sig_pos - 1.0                         # Calculo do erro (erro = previsão - alvo)

        neg_vecs = self.W_out[neg_idxs]     # Obtem vetores das palavras fora de contexto
        dots_neg = neg_vecs @ center_vec    # Similaridade entre o centro e cada amostra negativa
        errs_neg = sigmoid(dots_neg)        # Sigmoide → probabilidades previstas para os pares negativos

        # Gradiente do vetor central:
        # aproxima o contexto positivo e afasta os negativos
        grad_center = err_pos * ctx_vec + errs_neg @ neg_vecs

        # Ajusta vetor do contexto correto.
        self.W_out[context_idx] -= alpha * err_pos * center_vec
        # Afasta negativos.
        self.W_out[neg_idxs]    -= alpha * errs_neg[:, None] * center_vec[None, :]
        # Atualiza embedding da palavra central.
        self.W_in[center_idx]   -= alpha * grad_center

        # loss = -np.log(sig_pos + 1e-9) → sig_pos alto
        # -np.sum(np.log(sigmoid(-dots_neg) + 1e-9)) → negativos com similaridade baixa
        # 1e-9 evita log(0)
        loss = -np.log(sig_pos + 1e-9) - np.sum(np.log(sigmoid(-dots_neg) + 1e-9))

        return float(loss)

    # Negative samples
    def _sample_negatives(self, exclude_idx: int) -> np.ndarray:
        """
        Amostra palavras negativas para o Negative Sampling.

        As palavras são sorteadas usando a distribuição:
            P(w) ∝ freq(w)^0.75

        exclude_idx:
            índice da palavra positiva que deve ser evitada.
        """
        candidates = np.random.choice(
            self._vocab_indices,        # Índice completo do vocabulario
            size=self.neg_samples * 2,  # gera candidatos extras para garantir que sobrem negativos suficientes
            p=self.noise_dist           # Distribuição probabilística: P(w)∝freq(w)^0.75
        )
        candidates = candidates[candidates != exclude_idx] # Filtra o índice proibido.
        return candidates[: self.neg_samples] # # Retorna apenas a quantidade desejada de negativos

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

    # === TREINAMENTO ==========

    def train(
        self,
        tokens: list[str],
        epochs: int = 5,
        monitor_pairs: list[tuple[str, str]] | None = None,
    ) -> None:

        # Transforma tokens em índices
        indexed = [self.token2idx[t] for t in tokens if t in self.token2idx]

        n = len(indexed)            # Quantidade de tokens indexados
        total_steps = epochs * n    # Número de passos por epochs → decaimento de learning_rate
        step = 0                    # Contador de passos globais

        log.info("Treino: %d epocas x %d tokens = %d passos", epochs, n, total_steps)
        t_start = time.time()

        # Loop de épocas
        for epoch in range(epochs):
            epoch_loss, epoch_updates = 0.0, 0  # inicialização das estatíticas
            t_epoch = time.time()               # registrando início da época 

            # Loop dos índices dos tokens do corpora → o token da vez vira uma palavra central
            for i, center_idx in enumerate(indexed):
                alpha = self._current_alpha(step / total_steps) # Decaimento linear do learning_rate
                step += 1                                       # Incrementa o passo
                win = random.randint(1, self.window)            # Janela dinâmica

                # Analisando as win palavras à esquerda ou direita da palavra central
                for j in range(-win, win + 1):
                    # Ignora o próprio centro
                    if j == 0:
                        continue
                    ci = i + j # índice do contexto
                    # Segurança de limite do corpora
                    if ci < 0 or ci >= n:
                        continue
                    context_idx = indexed[ci]                       # palavra de contexto
                    neg_idxs = self._sample_negatives(context_idx)  # seleciona samples negatives
                    epoch_loss += self._sgd_update(center_idx, context_idx, neg_idxs, alpha) # atualiza
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