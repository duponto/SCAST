import argparse
import sys
from engine import (
    TextPreprocessor,
    TextVectorizer,
    SimilarityMetrics,
    CorrelationEngine,
    DocumentManager
)

def run_cli():
    parser = argparse.ArgumentParser(
        description="SCAST - Sistema de Comparação e Análise de Similaridade Textual",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python cli.py --file1 doc1.txt --file2 doc2.txt
  python cli.py --text1 "Olá mundo" --text2 "Olá a todos" --lang portuguese
  python cli.py --file1 doc1.txt --file2 doc2.txt --w-cos 0.7 --w-jac 0.2 --w-euc 0.1
        """
    )
    
    # Entradas por arquivos
    parser.add_argument("--file1", type=str, help="Caminho do primeiro arquivo de texto (.txt)")
    parser.add_argument("--file2", type=str, help="Caminho do segundo arquivo de texto (.txt)")
    
    # Entradas diretas por texto
    parser.add_argument("--text1", type=str, help="Texto direto 1")
    parser.add_argument("--text2", type=str, help="Texto direto 2")
    
    # Configurações de NLP e Pesos
    parser.add_argument("--lang", type=str, choices=["portuguese", "english"], default="portuguese",
                        help="Idioma do texto para processamento de stopwords e stemming (padrão: portuguese)")
    parser.add_argument("--no-stemming", action="store_true", help="Desativa o processo de stemming (normalização de radicais)")
    parser.add_argument("--no-stopwords", action="store_true", help="Desativa a remoção de stopwords")
    
    # Pesos da correlação
    parser.add_argument("--w-cos", type=float, default=0.5, help="Peso da Similaridade de Cosseno (padrão: 0.5)")
    parser.add_argument("--w-jac", type=float, default=0.3, help="Peso da Similaridade de Jaccard (padrão: 0.3)")
    parser.add_argument("--w-euc", type=float, default=0.2, help="Peso da Similaridade Euclidiana (padrão: 0.2)")

    args = parser.parse_args()
    
    # Validação de entradas
    t1, t2 = None, None
    source_name1, source_name2 = "Texto 1", "Texto 2"
    
    if args.file1 and args.file2:
        try:
            t1 = DocumentManager.read_file(args.file1)
            t2 = DocumentManager.read_file(args.file2)
            source_name1 = args.file1
            source_name2 = args.file2
        except Exception as e:
            print(f"Erro ao carregar arquivos: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.text1 and args.text2:
        t1 = args.text1
        t2 = args.text2
    else:
        parser.print_help()
        print("\n[Erro] Você deve fornecer dois arquivos (--file1 e --file2) ou dois textos diretos (--text1 e --text2).", file=sys.stderr)
        sys.exit(1)
        
    # Inicializa pipeline de NLP
    preprocessor = TextPreprocessor(
        language=args.lang,
        remove_stopwords=not args.no_stopwords,
        use_stemming=not args.no_stemming
    )
    
    # Pré-processa os textos
    tokens1 = preprocessor.preprocess(t1)
    tokens2 = preprocessor.preprocess(t2)
    
    prep_str1 = preprocessor.preprocess_as_string(t1)
    prep_str2 = preprocessor.preprocess_as_string(t2)
    
    # Inicializa vetorizador estatístico e converte em vetores TF-IDF
    vectorizer = TextVectorizer(use_sublinear_tf=True)
    # Fit no corpus composto pelos dois textos
    try:
        vectorizer.fit([prep_str1, prep_str2])
        vec1 = vectorizer.transform(prep_str1)
        vec2 = vectorizer.transform(prep_str2)
    except Exception as e:
        print(f"Erro no processo de vetorização: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Cálculo das Métricas de Similaridade
    cos_score = SimilarityMetrics.cosine_similarity(vec1, vec2)
    jac_score = SimilarityMetrics.jaccard_similarity(tokens1, tokens2)
    euc_score = SimilarityMetrics.normalized_euclidean_similarity(vec1, vec2)
    
    # Consolidação do Índice de Correlação
    weights = {
        'cosine': args.w_cos,
        'jaccard': args.w_jac,
        'euclidean': args.w_euc
    }
    
    results = CorrelationEngine.calculate_correlation(
        cosine_score=cos_score,
        jaccard_score=jac_score,
        euclidean_score=euc_score,
        weights=weights
    )
    
    # Exibe o Relatório Formatado no Terminal
    print("\n" + "="*60)
    print(" SCAST - RELATORIO DE ANALISE DE SIMILARIDADE TEXTUAL")
    print("="*60)
    print(f"Documento 1: {source_name1} ({len(t1)} caracteres)")
    print(f"Documento 2: {source_name2} ({len(t2)} caracteres)")
    print(f"Idioma: {args.lang.upper()} | Stemming: {'Ativo' if not args.no_stemming else 'Inativo'} | Stopwords: {'Removidas' if not args.no_stopwords else 'Mantidas'}")
    print("-"*60)
    print("METRICAS INDIVIDUAIS:")
    print(f"  - Similaridade de Cosseno:   {results['scores']['cosine']:.4f} (Peso: {results['weights']['cosine']:.2f})")
    print(f"  - Similaridade de Jaccard:   {results['scores']['jaccard']:.4f} (Peso: {results['weights']['jaccard']:.2f})")
    print(f"  - Similaridade Euclidiana:   {results['scores']['euclidean']:.4f} (Peso: {results['weights']['euclidean']:.2f})")
    print("-"*60)
    print("INDICE DE CORRELACAO CONSOLIDADO:")
    print(f"  -> Indice final:   {results['correlation_index']:.4f} ({results['correlation_index']*100:.1f}%)")
    print(f"  -> Interpretacao:  {results['interpretation']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_cli()
