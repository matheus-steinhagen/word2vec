"""
Word2Vec — Skip-gram + Negative Sampling (SGNS)
"""

from __future__ import annotations

import logging

from pre_processing import(
    load_corpus,
    tokenize,
    build_vocabulary,
    subsample_tokens,
)

from model import Word2Vec

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def main() -> None:

    # === INICIAR VARIÁVEIS ==========

    CORPUS_PATH = "./corpus.txt"
    SAVE_PATH   = "./embeddings.json"

    DIM         = 50    # 300 so vale com >10M tokens; aqui causa overfitting
    WINDOW      = 5
    NEG_SAMPLES = 5     # acima de 10 prejudica; 5 e o padrao do paper
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

    # === PRÉ PROCESSAMENTO ==========

    # Carregar corpus
    corpus = load_corpus(CORPUS_PATH)

    # Tokenizar
    tokens_raw = tokenize(corpus)

    # Remover stopwords
    tokens_clean = [t for t in tokens_raw if t not in STOPWORDS]

    # Construir vocabulario
    vocab, token2idx, counts = build_vocabulary(tokens_clean, min_count=MIN_COUNT)

    # Filtra OOV (out of vocabulary)
    tokens_filtered = [t for t in tokens_clean if t in token2idx]

    # Subsampling
    tokens = subsample_tokens(tokens_filtered, counts, len(tokens_filtered), t=T_SUBSAMPLE)

    # === PROCESSAMENTO ==========

    # Inicializando modelo
    model = Word2Vec(
        vocab,
        token2idx,
        counts,
        dim=DIM,
        window=WINDOW,
        neg_samples=NEG_SAMPLES,
        alpha=ALPHA,
        min_alpha=MIN_ALPHA,
    )

    # Treinando modelo
    model.train(
        tokens,
        epochs=EPOCHS,
        monitor_pairs=MONITOR_PAIRS
    )
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

        # Chamando o word2vec para treino
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