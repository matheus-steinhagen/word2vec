# Meu primeiro modelo de vetorização de palavras

## Passos de trabalho
1. tokenizar o corpora (tokens)
2. listar tokens únicos (vocabulario)
3. gerar embeddings: main, left e right

4. varrer (corpora_tokens)
    1. localizar palavra central (main_token)
    2. localizar palavra à esquerda (left_token)
        2.2. registrar palavra no dicionario [main_token].push(left_token)
    3. localizar embeddings[left_token]
    4. filtrar left_embedding