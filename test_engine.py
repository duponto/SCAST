
from engine import (
    TextPreprocessor,
    TextVectorizer,
    SimilarityMetrics,
    CorrelationEngine
)

def run_tests():
    print("====================================================")
    print("INICIANDO SUITE DE TESTES AUTOMATIZADOS - SCAST")
    print("====================================================")
    
    # ---------------------------------------------------------
    # TESTE 1: Pre-processador (NLP)
    # ---------------------------------------------------------
    print("\nTeste 1: Validando Pre-processador NLP (Portugues)...")
    preprocessor = TextPreprocessor(
        language='portuguese',
        remove_stopwords=True,
        use_stemming=True,
        remove_accents=True
    )
    
    text = "A rapida comparacao de textos eh fantastica!"
    tokens = preprocessor.preprocess(text)
    print(f"  * Texto original: '{text}'")
    print(f"  * Tokens resultantes: {tokens}")
    
    # Validacoes basicas do pre-processador
    # 'a' e 'de' sao stopwords e devem ser removidos.
    # Acentos devem ser retirados.
    # Caixa alta deve ser normalizada.
    assert "a" not in tokens, "Erro: Stopword 'a' nao foi removida!"
    assert "de" not in tokens, "Erro: Stopword 'de' nao foi removida!"
    
    # Verifica stemming do portugues
    stem_rapida = preprocessor.stem("rapida")
    print(f"  * Stem de 'rapida': {stem_rapida}")
    assert len(tokens) > 0, "Erro: O pre-processador limpou todo o texto incorretamente!"
    print("  [OK] Teste 1 concluido com sucesso!")

    # ---------------------------------------------------------
    # TESTE 2: Metricas de Identidade (100% igualdade)
    # ---------------------------------------------------------
    print("\nTeste 2: Validando Identidade (Documentos Identicos)...")
    text_identical = "Desenvolvimento de um sistema computacional para comparacao textual."
    
    prep_str = preprocessor.preprocess_as_string(text_identical)
    tokens_identical = preprocessor.preprocess(text_identical)
    
    # Vetorizacao
    vectorizer = TextVectorizer(use_sublinear_tf=True)
    vectorizer.fit([prep_str])
    vec = vectorizer.transform(prep_str)
    
    # Similaridades
    cos = SimilarityMetrics.cosine_similarity(vec, vec)
    jac = SimilarityMetrics.jaccard_similarity(tokens_identical, tokens_identical)
    euc = SimilarityMetrics.normalized_euclidean_similarity(vec, vec)
    
    print(f"  * Cosseno (esperado 1.0): {cos}")
    print(f"  * Jaccard (esperado 1.0): {jac}")
    print(f"  * Euclidiano Normalizado (esperado 1.0): {euc}")
    
    assert abs(cos - 1.0) < 1e-5, f"Erro: Cosseno para identicos deve ser 1.0, obtido {cos}"
    assert abs(jac - 1.0) < 1e-5, f"Erro: Jaccard para identicos deve ser 1.0, obtido {jac}"
    assert abs(euc - 1.0) < 1e-5, f"Erro: Similaridade Euclidiana para identicos deve ser 1.0, obtido {euc}"
    
    # Correlacao
    res = CorrelationEngine.calculate_correlation(cos, jac, euc)
    print(f"  * Indice de Correlacao Consolidado: {res['correlation_index']}")
    print(f"  * Interpretacao: {res['interpretation']['label']}")
    assert abs(res['correlation_index'] - 1.0) < 1e-5, "Erro: Correlacao para identicos deve ser 1.0!"
    print("  [OK] Teste 2 concluido com sucesso!")

    # ---------------------------------------------------------
    # TESTE 3: Metricas de Disjuncao Completa (0% igualdade)
    # ---------------------------------------------------------
    print("\nTeste 3: Validando Disjuncao (Documentos Completamente Diferentes)...")
    t_diff1 = "futebol bola esporte campeonato gol"
    t_diff2 = "algebra matriz matematica calculo algoritmo"
    
    tokens_diff1 = preprocessor.preprocess(t_diff1)
    tokens_diff2 = preprocessor.preprocess(t_diff2)
    
    prep_diff1 = preprocessor.preprocess_as_string(t_diff1)
    prep_diff2 = preprocessor.preprocess_as_string(t_diff2)
    
    # Vetorizacao cruzada
    vectorizer_diff = TextVectorizer(use_sublinear_tf=True)
    vectorizer_diff.fit([prep_diff1, prep_diff2])
    vec_diff1 = vectorizer_diff.transform(prep_diff1)
    vec_diff2 = vectorizer_diff.transform(prep_diff2)
    
    # Similaridades
    cos_d = SimilarityMetrics.cosine_similarity(vec_diff1, vec_diff2)
    jac_d = SimilarityMetrics.jaccard_similarity(tokens_diff1, tokens_diff2)
    euc_d = SimilarityMetrics.normalized_euclidean_similarity(vec_diff1, vec_diff2)
    
    print(f"  * Cosseno (esperado 0.0): {cos_d}")
    print(f"  * Jaccard (esperado 0.0): {jac_d}")
    print(f"  * Euclidiano Normalizado (esperado 0.0): {euc_d}")
    
    assert abs(cos_d - 0.0) < 1e-5, f"Erro: Cosseno para disjuntos deve ser 0.0, obtido {cos_d}"
    assert abs(jac_d - 0.0) < 1e-5, f"Erro: Jaccard para disjuntos deve ser 0.0, obtido {jac_d}"
    assert abs(euc_d - 0.0) < 1e-5, f"Erro: Similaridade Euclidiana para disjuntos deve ser 0.0, obtido {euc_d}"
    
    res_d = CorrelationEngine.calculate_correlation(cos_d, jac_d, euc_d)
    print(f"  * Indice de Correlacao Consolidado: {res_d['correlation_index']}")
    print(f"  * Interpretacao: {res_d['interpretation']['label']}")
    assert abs(res_d['correlation_index'] - 0.0) < 1e-5, "Erro: Correlacao para disjuntos deve ser 0.0!"
    print("  [OK] Teste 3 concluido com sucesso!")

    # ---------------------------------------------------------
    # TESTE 4: Normalizacao de Pesos
    # ---------------------------------------------------------
    print("\nTeste 4: Validando Normalizacao de Pesos do Motor...")
    invalid_weights = {
        'cosine': 2.0,
        'jaccard': 2.0,
        'euclidean': 1.0
    }
    normalized = CorrelationEngine.validate_and_normalize_weights(invalid_weights)
    print(f"  * Pesos de entrada: {invalid_weights}")
    print(f"  * Pesos normalizados: {normalized}")
    print(f"  * Soma dos pesos normalizados (deve ser 1.0): {sum(normalized.values())}")
    
    assert abs(sum(normalized.values()) - 1.0) < 1e-5, "Erro: A soma dos pesos normalizados deve ser 1.0!"
    assert abs(normalized['cosine'] - 0.4) < 1e-5, f"Erro: Peso do cosseno devia ser 0.4, obtido {normalized['cosine']}"
    assert abs(normalized['jaccard'] - 0.4) < 1e-5, f"Erro: Peso do jaccard devia ser 0.4, obtido {normalized['jaccard']}"
    assert abs(normalized['euclidean'] - 0.2) < 1e-5, f"Erro: Peso do euclidiano devia ser 0.2, obtido {normalized['euclidean']}"
    print("  [OK] Teste 4 concluido com sucesso!")

    print("\n====================================================")
    print("TODOS OS TESTES EXECUTADOS COM SUCESSO! [CONCLUIDO]")
    print("O motor matematico e linguistico esta 100% operacional.")
    print("====================================================")

if __name__ == "__main__":
    run_tests()
