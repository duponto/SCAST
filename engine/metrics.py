import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances as sklearn_euclidean_distances

class SimilarityMetrics:
    """
    Classe utilitária contendo implementações estáticas para o cálculo
    das diferentes métricas de similaridade entre representações textuais.
    """
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Calcula a similaridade do cosseno entre dois vetores.
        
        :param vec1: Vetor esparso TF-IDF do primeiro documento.
        :param vec2: Vetor esparso TF-IDF do segundo documento.
        :return: Valor de similaridade no intervalo [0.0, 1.0].
        """
        try:
            # sklearn_cosine_similarity suporta matrizes esparsas diretamente
            score = sklearn_cosine_similarity(vec1, vec2)[0][0]
            # Limita a precisão numérica entre 0.0 e 1.0 para evitar imprecisões de ponto flutuante
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def jaccard_similarity(tokens1, tokens2):
        """
        Calcula a similaridade de Jaccard entre dois conjuntos de tokens pré-processados.
        Mapeia a interseção dividida pela união do vocabulário literal.
        
        :param tokens1: Lista ou set de tokens do primeiro documento.
        :param tokens2: Lista ou set de tokens do segundo documento.
        :return: Valor de similaridade no intervalo [0.0, 1.0].
        """
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Caso especial: Ambos os documentos estão vazios após processamento
        if not set1 and not set2:
            return 1.0  # Documentos vazios são idênticos entre si
            
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        score = len(intersection) / len(union)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def normalized_euclidean_similarity(vec1, vec2):
        """
        Calcula a Similaridade Euclidiana Normalizada baseada na Distância Euclidiana.
        Como os vetores são normalizados em norma L2 (comprimento unitário = 1),
        a distância euclidiana máxima d é sqrt(2) ≈ 1.4142.
        
        Fórmula Proposta: Similarity = 1 - (d / sqrt(2))
        
        :param vec1: Vetor esparso TF-IDF do primeiro documento.
        :param vec2: Vetor esparso TF-IDF do segundo documento.
        :return: Valor de similaridade no intervalo [0.0, 1.0].
        """
        try:
            # Calcula a distância euclidiana geométrica
            dist = sklearn_euclidean_distances(vec1, vec2)[0][0]
            
            # Distância máxima teórica para vetores L2 na hiperesfera unitária
            max_dist = np.sqrt(2)
            
            # Normalização para transformar distância (dissimilaridade) em similaridade
            score = 1.0 - (dist / max_dist)
            
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return 0.0
