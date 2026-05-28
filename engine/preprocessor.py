import re
import unicodedata
import sys

from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.stem.snowball import SnowballStemmer

class TextPreprocessor:
    """
    Classe responsável pelo pré-processamento de textos.
    Realiza tokenização, conversão para minúsculas, remoção de acentos,
    filtragem de stopwords e redução das palavras ao seu radical (stemming).
    """
    
    # Listas locais de stopwords como fallback de segurança caso o NLTK falhe
    PORTUGUESE_STOPWORDS_FALLBACK = {
        'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até',
        'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do',
        'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'essa',
        'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes', 'estou', 'eu', 'foi',
        'fomos', 'foram', 'fosse', 'fui', 'há', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais',
        'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'nas', 'nem',
        'no', 'nos', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 'numa', 'o', 'os', 'ou',
        'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem',
        'se', 'seja', 'sejam', 'sem', 'ser', 'será', 'serão', 'seu', 'seus', 'só', 'sob',
        'sobre', 'sua', 'suas', 'também', 'te', 'tem', 'temos', 'tenho', 'ter', 'teu',
        'teus', 'tu', 'tua', 'tuas', 'um', 'uma', 'umas', 'uns', 'vos', 'você', 'vocês'
    }
    
    ENGLISH_STOPWORDS_FALLBACK = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a',
        'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at',
        'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on',
        'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
        'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
        "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
        'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
        'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
        'won', "won't", 'wouldn', "wouldn't"
    }

    def __init__(self, language='portuguese', remove_stopwords=True, use_stemming=True, remove_accents=True):
        """
        Inicializa o pré-processador com opções customizáveis.
        
        :param language: 'portuguese' ou 'english'
        :param remove_stopwords: Se True, remove as palavras funcionais (stopwords)
        :param use_stemming: Se True, reduz os termos aos radicais
        :param remove_accents: Se True, remove acentos dos caracteres (ex: 'á' -> 'a')
        """
        self.language = language.lower()
        if self.language not in ['portuguese', 'english']:
            self.language = 'portuguese'
            
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.remove_accents = remove_accents
        
        # Inicializa recursos de Stopwords
        self.stopwords = self._load_stopwords()
        
        # Inicializa Stemmer de acordo com o idioma
        self.stemmer = self._init_stemmer()

    def _load_stopwords(self):
        """Carrega a lista de stopwords do NLTK ou usa a lista local caso ocorra erro."""
        if not self.remove_stopwords:
            return set()
            
        try:
            # Tenta carregar do NLTK
            return set(stopwords.words(self.language))
        except Exception:
            # Fallback seguro local caso o NLTK não esteja disponível no momento
            print(f"[Aviso] Falha ao carregar stopwords do NLTK para '{self.language}'. Utilizando fallback local.", file=sys.stderr)
            if self.language == 'portuguese':
                return self.PORTUGUESE_STOPWORDS_FALLBACK
            else:
                return self.ENGLISH_STOPWORDS_FALLBACK

    def _init_stemmer(self):
        """Inicializa o stemmer correspondente ao idioma configurado."""
        if not self.use_stemming:
            return None
            
        try:
            if self.language == 'portuguese':
                return RSLPStemmer()
            else:
                return SnowballStemmer("english")
        except Exception as e:
            print(f"[Erro] Não foi possível carregar o Stemmer para '{self.language}': {e}. Stemming desativado.", file=sys.stderr)
            return None

    @staticmethod
    def strip_accents(text):
        """Remove acentuações e diacríticos de uma string."""
        if text is None:
            return ""
        nfkd_form = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

    def tokenize(self, text):
        """
        Tokeniza o texto convertendo para caixa baixa e limpando pontuações.
        Usa expressão regular robusta para lidar com pontuação de forma performática.
        """
        if not text:
            return []
            
        # 1. Caixa baixa
        text = text.lower()
        
        # 2. Remoção de acentos (se configurado)
        if self.remove_accents:
            text = self.strip_accents(text)
            
        # 3. Limpeza de caracteres não alfanuméricos preservando espaços
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # 4. Tokenização simples por espaços em branco
        tokens = text.split()
        
        # 5. Filtragem de tokens vazios ou puramente pontuações/hifens
        tokens = [t for t in tokens if t.strip() and t != '-' and not t.isdigit()]
        
        return tokens

    def stem(self, word):
        """Aplica o stemming em uma palavra individual se o stemmer estiver ativo."""
        if not self.stemmer or not word:
            return word
        try:
            return self.stemmer.stem(word)
        except Exception:
            # Fallback seguro: retorna a própria palavra se o stemmer gerar erro
            return word

    def preprocess(self, text):
        """
        Executa o pipeline completo de pré-processamento no texto.
        
        :param text: String contendo o texto original.
        :return: Lista de tokens pré-processados.
        """
        if not text:
            return []
            
        # Tokenização e padronização (lowercase, acentos)
        tokens = self.tokenize(text)
        
        # Remoção de Stopwords
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
            
        # Aplicação de Stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stem(t) for t in tokens]
            
        # Filtragem final de strings vazias
        tokens = [t for t in tokens if t.strip()]
        
        return tokens

    def preprocess_as_string(self, text):
        """
        Executa o pré-processamento e junta os tokens em uma única string espacada.
        Útil para alimentar o vetorizador TF-IDF do scikit-learn.
        """
        tokens = self.preprocess(text)
        return " ".join(tokens)
