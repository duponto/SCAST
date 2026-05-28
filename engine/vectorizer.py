from sklearn.feature_extraction.text import TfidfVectorizer


class TextVectorizer:
    """
    Classe responsável pela representação vetorial dos textos.
    Utiliza o modelo TF-IDF (Term Frequency-Inverse Document Frequency)
    com suporte para N-Grams e sublinear TF scaling.
    """
    
    def __init__(self, use_sublinear_tf=True, ngram_range=(1, 2), analyzer='word'):
        """
        Inicializa o vetorizador com opções de modelagem estatística.
        
        :param use_sublinear_tf: Se True, aplica escala logarítmica à frequência (1 + log(tf))
                                 para amortecer o peso de termos excessivamente repetidos.
        :param ngram_range: Tupla (min_n, max_n) definindo a faixa de N-grams.
        :param analyzer: 'word' para n-grams de palavras, 'char' para n-grams de caracteres.
        """
        self.use_sublinear_tf = use_sublinear_tf
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        
        # Inicializa o TfidfVectorizer do scikit-learn
        # Usamos norm='l2' para que a distância euclidiana seja delimitada e normalizável
        self.vectorizer = TfidfVectorizer(
            norm='l2',
            sublinear_tf=self.use_sublinear_tf,
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            token_pattern=r'\S+'  # Como o texto já é pré-processado, os tokens são separados por espaço
        )
        self.is_fitted = False

    def fit(self, preprocessed_texts):
        """
        Ajusta o vocabulário e os pesos IDF com base em um corpus de textos pré-processados.
        
        :param preprocessed_texts: Lista de strings, onde cada string é um texto já tokenizado e limpo.
        :return: A própria instância (self).
        """
        if not preprocessed_texts:
            raise ValueError("O corpus fornecido para ajuste (fit) está vazio.")
            
        # Filtra strings vazias para evitar problemas com vetorizadores
        clean_texts = [text for text in preprocessed_texts if text.strip()]
        if not clean_texts:
            # Fallback caso todas as strings estejam vazias
            clean_texts = ["documento_vazio"]
            
        self.vectorizer.fit(clean_texts)
        self.is_fitted = True
        return self

    def transform(self, preprocessed_texts):
        """
        Transforma os textos pré-processados em vetores de pesos TF-IDF.
        
        :param preprocessed_texts: Lista de strings ou string individual a ser convertida.
        :return: Matriz esparsa de vetores (ou vetor único) em formato NumPy/Scipy.
        """
        if not self.is_fitted:
            # Se não foi feito o fit antes, fazemos o fit_transform para evitar travamentos
            self.fit(preprocessed_texts)
            
        if isinstance(preprocessed_texts, str):
            preprocessed_texts = [preprocessed_texts]
            
        # Garante que textos completamente vazios sejam tratados como strings normais
        clean_texts = [text if text.strip() else "" for text in preprocessed_texts]
        
        return self.vectorizer.transform(clean_texts)

    def fit_transform(self, preprocessed_texts):
        """
        Ajusta o vocabulário e transforma o corpus em uma única operação.
        """
        self.fit(preprocessed_texts)
        return self.transform(preprocessed_texts)

    def get_feature_names(self):
        """Retorna a lista de termos/n-grams que compõem as dimensões do vetor."""
        if not self.is_fitted:
            return []
        return self.vectorizer.get_feature_names_out()

    def get_vocabulary(self):
        """Retorna o dicionário de mapeamento termo -> índice."""
        if not self.is_fitted:
            return {}
        return self.vectorizer.vocabulary_
