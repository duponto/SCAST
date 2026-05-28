
class CorrelationEngine:
    """
    Classe responsável por fundir as métricas individuais de similaridade
    em um único Índice de Correlação Consolidado por meio de uma combinação ponderada.
    """
    
    DEFAULT_WEIGHTS = {
        'cosine': 0.5,
        'jaccard': 0.3,
        'euclidean': 0.2
    }

    @classmethod
    def validate_and_normalize_weights(cls, weights):
        """
        Valida se os pesos fornecidos são válidos e os normaliza
        para garantir que a soma seja exatamente igual a 1.0.
        
        :param weights: Dicionário contendo pesos para 'cosine', 'jaccard' e 'euclidean'.
        :return: Dicionário de pesos validados e normalizados.
        """
        if not weights or not isinstance(weights, dict):
            return cls.DEFAULT_WEIGHTS.copy()
            
        validated = {}
        for key in ['cosine', 'jaccard', 'euclidean']:
            val = weights.get(key, 0.0)
            try:
                # Converte para float positivo
                validated[key] = max(0.0, float(val))
            except (ValueError, TypeError):
                validated[key] = 0.0
                
        total_sum = sum(validated.values())
        
        # Caso a soma de todos os pesos seja 0, redistribui igualmente
        if total_sum == 0:
            return {k: 1.0 / 3.0 for k in ['cosine', 'jaccard', 'euclidean']}
            
        # Normalização dos pesos para somar 1.0
        normalized = {k: v / total_sum for k, v in validated.items()}
        return normalized

    @classmethod
    def interpret_index(cls, index):
        """
        Fornece uma interpretação qualitativa e contextualizada do Índice de Correlação.
        A escala é projetada para refletir que mesmo valores baixos de TF-IDF podem
        indicar relação temática real entre textos parafraseados.
        
        :param index: Valor numérico contido no intervalo [0.0, 1.0].
        :return: Dicionário com 'label', 'description' e 'color'.
        """
        if index >= 1.0:
            return {
                'label': 'Correspondência Idêntica',
                'description': 'Os textos são idênticos ou praticamente iguais em conteúdo e vocabulário.',
                'color': '#00e1d9',
                'icon': '🔷'
            }
        elif index >= 0.75:
            return {
                'label': 'Correspondência Muito Alta',
                'description': 'Os textos compartilham a grande maioria do vocabulário e estrutura. Provável reprodução com pequenas alterações.',
                'color': '#00e1d9',
                'icon': '🔷'
            }
        elif index >= 0.55:
            return {
                'label': 'Correspondência Alta',
                'description': 'Forte sobreposição vocabular. Os textos possivelmente derivam da mesma fonte ou tratam do mesmo assunto com linguagem similar.',
                'color': '#26a69a',
                'icon': '🟢'
            }
        elif index >= 0.35:
            return {
                'label': 'Correspondência Moderada',
                'description': 'Sobreposição significativa de termos. Os textos provavelmente abordam o mesmo tema, embora com vocabulário parcialmente distinto.',
                'color': '#66bb6a',
                'icon': '🟡'
            }
        elif index >= 0.20:
            return {
                'label': 'Relação Temática Detectada',
                'description': 'Os textos compartilham termos-chave, sugerindo assunto em comum. As diferenças vocabulares reduzem a pontuação estatística, mas a relação temática é perceptível.',
                'color': '#ffa726',
                'icon': '🟠'
            }
        elif index >= 0.08:
            return {
                'label': 'Relação Temática Parcial',
                'description': 'Há indícios de vocabulário compartilhado. Os textos podem abordar temas próximos utilizando linguagem bastante distinta (paráfrase). Consulte o realce visual para validar.',
                'color': '#ffb74d',
                'icon': '🔸'
            }
        elif index > 0.0:
            return {
                'label': 'Relação Vestigial',
                'description': 'Pouquíssimos termos em comum. Os textos podem ter alguma sobreposição temática marginal, mas o vocabulário é predominantemente distinto.',
                'color': '#b0bec5',
                'icon': '⚪'
            }
        else:
            return {
                'label': 'Sem Relação Detectável',
                'description': 'Nenhum termo significativo compartilhado entre os textos. Os documentos tratam de assuntos completamente distintos.',
                'color': '#78909c',
                'icon': '⚫'
            }

    @classmethod
    def calculate_lexical_coverage(cls, tokens1, tokens2, preprocessor=None):
        """
        Calcula a cobertura lexical complementar entre dois textos,
        incluindo correspondências por radical (stem), oferecendo uma
        perspectiva mais humana que complementa as métricas matemáticas.
        
        :param tokens1: Lista de tokens brutos (sem stemming) do texto 1.
        :param tokens2: Lista de tokens brutos (sem stemming) do texto 2.
        :param preprocessor: Instância do TextPreprocessor (para stemming).
        :return: Dicionário com métricas de cobertura.
        """
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Correspondências literais exatas
        exact_matches = set1 & set2
        
        # Correspondências por radical (stem)
        stem_matches_1 = set()
        stem_matches_2 = set()
        if preprocessor and preprocessor.use_stemming:
            stems1 = {t: preprocessor.stem(t) for t in set1}
            stems2 = {t: preprocessor.stem(t) for t in set2}
            stem_set1 = set(stems1.values())
            stem_set2 = set(stems2.values())
            shared_stems = stem_set1 & stem_set2
            
            # Tokens que matcham por stem mas NÃO são exatos
            for t in set1:
                if t not in exact_matches and stems1[t] in shared_stems:
                    stem_matches_1.add(t)
            for t in set2:
                if t not in exact_matches and stems2[t] in shared_stems:
                    stem_matches_2.add(t)
        
        total_unique = len(set1 | set2)
        total_related = len(exact_matches) + max(len(stem_matches_1), len(stem_matches_2))
        
        coverage_ratio = total_related / total_unique if total_unique > 0 else 0.0
        
        return {
            'exact_matches': len(exact_matches),
            'stem_matches_doc1': len(stem_matches_1),
            'stem_matches_doc2': len(stem_matches_2),
            'total_unique_terms': total_unique,
            'coverage_ratio': coverage_ratio,
            'terms_doc1': len(set1),
            'terms_doc2': len(set2),
        }

    @classmethod
    def calculate_correlation(cls, cosine_score, jaccard_score, euclidean_score, weights=None):
        """
        Calcula o Índice de Correlação consolidado a partir das métricas individuais.
        
        :param cosine_score: Pontuação de similaridade do cosseno [0, 1]
        :param jaccard_score: Pontuação de similaridade de Jaccard [0, 1]
        :param euclidean_score: Pontuação de similaridade euclidiana normalizada [0, 1]
        :param weights: Dicionário opcional contendo os pesos das métricas.
        :return: Dicionário com pontuações detalhadas, índice final e interpretação.
        """
        norm_weights = cls.validate_and_normalize_weights(weights)
        
        # Fórmula ponderada: Sum(weight_i * score_i)
        correlation_index = (
            norm_weights['cosine'] * cosine_score +
            norm_weights['jaccard'] * jaccard_score +
            norm_weights['euclidean'] * euclidean_score
        )
        
        # Garante limites estritos [0.0, 1.0]
        correlation_index = min(1.0, max(0.0, float(correlation_index)))
        
        return {
            'scores': {
                'cosine': cosine_score,
                'jaccard': jaccard_score,
                'euclidean': euclidean_score
            },
            'weights': norm_weights,
            'correlation_index': correlation_index,
            'interpretation': cls.interpret_index(correlation_index)
        }
