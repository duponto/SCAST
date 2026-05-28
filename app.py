import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re

# Inicializar recursos NLTK (deve ser antes de qualquer outro import)
import nltk_setup

from engine import (
    TextPreprocessor,
    TextVectorizer,
    SimilarityMetrics,
    CorrelationEngine
)

# Configuração da página do Streamlit
st.set_page_config(
    page_title="SCAST - Análise de Similaridade Textual",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada para visual Premium e Moderno (Dark Mode / Neon Gradientes)
st.markdown("""
<style>
    /* Importa fonte Inter do Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Configura fonte global */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Estilo para Títulos Principais e Efeitos Hover */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00c6ff 0%, #0072ff 50%, #8a2be2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0px 4px 10px rgba(0, 114, 255, 0.15);
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Cards de Métricas Estilizados (Glassmorphism e Neon) */
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 198, 255, 0.6);
        box-shadow: 0 8px 30px rgba(0, 198, 255, 0.15);
    }
    
    .metric-title {
        color: #8b949e;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.25rem;
    }
    
    .metric-weight {
        font-size: 0.75rem;
        color: #00c6ff;
        font-weight: 500;
    }
    
    /* Destaques de Realce de Texto (Similarity Highlight badges) */
    .highlight-exact {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1.5px solid rgba(46, 160, 67, 0.6);
        color: #56d364;
        border-radius: 4px;
        padding: 1px 4px;
        font-weight: 500;
    }
    
    .highlight-stem {
        background-color: rgba(210, 153, 34, 0.15);
        border: 1.5px solid rgba(210, 153, 34, 0.6);
        color: #e3b341;
        border-radius: 4px;
        padding: 1px 4px;
        font-weight: 500;
    }
    
    .highlight-none {
        color: #c9d1d9;
    }
    
    /* Container para caixas de texto com realce */
    .text-highlight-box {
        background: rgba(13, 17, 23, 0.95);
        border: 1px solid rgba(48, 54, 61, 0.8);
        border-radius: 8px;
        padding: 1.25rem;
        max-height: 400px;
        overflow-y: auto;
        line-height: 1.8;
        font-size: 0.95rem;
        white-space: pre-wrap;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# FUNÇÃO DE REALCE DE SIMILARIDADE VISUAL
# ---------------------------------------------------------
def generate_visual_highlighting(text1, text2, preprocessor):
    """
    Gera as representações em HTML de ambos os textos realçando correspondências.
    Verde: Palavras idênticas literalmente.
    Amarelo: Palavras que compartilham o mesmo radical (Stem).
    """
    if not text1 or not text2:
        return text1, text2
        
    # Pré-processamento
    tokens1 = preprocessor.tokenize(text1)
    tokens2 = preprocessor.tokenize(text2)
    
    # Stemming
    stems1 = {t: preprocessor.stem(t) for t in tokens1}
    stems2 = {t: preprocessor.stem(t) for t in tokens2}
    
    set_tokens1_lower = {t.lower() for t in tokens1}
    set_tokens2_lower = {t.lower() for t in tokens2}
    
    set_stems1 = set(stems1.values())
    set_stems2 = set(stems2.values())
    
    # Função auxiliar para mapear palavras de forma segura para o HTML formatado
    def highlight_text(original_text, other_tokens_lower, other_stems, local_stems):
        # Quebra mantendo espaços, pontuações e quebras de linha
        words_and_spaces = re.split(r'(\s+|[^\w\s-])', original_text)
        
        html_out = []
        for item in words_and_spaces:
            if not item.strip():
                # Apenas espaço ou quebra de linha
                html_out.append(item)
                continue
                
            clean_word = re.sub(r'[^\w-]', '', item).lower()
            if not clean_word or clean_word == '-':
                html_out.append(item)  # Apenas pontuação
                continue
                
            # Normalização de acentos para busca assertiva de correspondências
            lookup_word = clean_word
            if preprocessor.remove_accents:
                lookup_word = preprocessor.strip_accents(clean_word)
                
            # Se for stopword (e remoção estiver ativa), escreve sem realce para evitar ruídos visuais
            if remove_stopwords and lookup_word in preprocessor.stopwords:
                html_out.append(item)
                continue
                
            # Mapeamento do radical (stem) com base no token normalizado
            word_stem = local_stems.get(lookup_word, preprocessor.stem(lookup_word))
            
            if lookup_word in other_tokens_lower:
                # Correspondência Literal Exata -> Verde
                html_out.append(f'<span class="highlight-exact">{item}</span>')
            elif word_stem in other_stems:
                # Correspondência de Radical Semântico -> Amarelo
                html_out.append(f'<span class="highlight-stem">{item}</span>')
            else:
                # Sem correspondência -> Padrão
                html_out.append(item)
                
        return "".join(html_out)
        
    html1 = highlight_text(text1, set_tokens2_lower, set_stems2, stems1)
    html2 = highlight_text(text2, set_tokens1_lower, set_stems1, stems2)
    
    return html1, html2

# ---------------------------------------------------------
# BARRA LATERAL (CONFIGURAÇÕES DO MOTOR)
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Painel de Configurações")

# Seleção de Idioma
language = st.sidebar.selectbox(
    "Idioma do Pipeline de NLP",
    options=["Português", "English"],
    index=0,
    help="Define o dicionário de stopwords e o modelo de stemming a ser utilizado."
)
lang_code = 'portuguese' if language == "Português" else 'english'

# Configurações de Pré-processamento
st.sidebar.markdown("#### Pré-processamento")
remove_stopwords = st.sidebar.checkbox("Remover Stopwords", value=True, help="Elimina palavras com pouco valor semântico (artigos, preposições, etc.)")
use_stemming = st.sidebar.checkbox("Aplicar Stemming", value=True, help="Reduz palavras às suas formas base/radical (ex: comparando, comparação -> comparar)")
remove_accents = st.sidebar.checkbox("Ignorar Acentos", value=True, help="Normaliza caracteres diacríticos (ex: á -> a) para evitar distorções ortográficas")

# Configurações do Vetorizador TF-IDF
st.sidebar.markdown("#### Parâmetros TF-IDF")
sublinear_tf = st.sidebar.checkbox("Sublinear TF Scaling", value=True, help="Utiliza escala logarítmica (1 + log(tf)) para amenizar termos super-repetidos.")
ngram_option = st.sidebar.selectbox(
    "Faixa de N-Grams",
    options=["Apenas Palavras Únicas (1, 1)", "Palavras Únicas e Bigramas (1, 2)", "Apenas Bigramas (2, 2)"],
    index=1
)
ngram_range = (1, 1) if "1, 1" in ngram_option else ((1, 2) if "1, 2" in ngram_option else (2, 2))

# Configurações de Pesos para o Índice de Correlação
st.sidebar.markdown("#### Pesos do Índice de Correlação")
st.sidebar.caption("Ajuste a influência de cada métrica no resultado consolidado. A soma será normalizada automaticamente para 1.0.")
w_cos = st.sidebar.slider("Peso do Cosseno", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
w_jac = st.sidebar.slider("Peso do Jaccard", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
w_euc = st.sidebar.slider("Peso do Euclidiano Normalizado", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

user_weights = {
    'cosine': w_cos,
    'jaccard': w_jac,
    'euclidean': w_euc
}

# ---------------------------------------------------------
# CORPO PRINCIPAL DO APP
# ---------------------------------------------------------
st.markdown('<div class="main-title">SCAST</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Sistema de Comparação e Análise de Similaridade entre Conteúdos Textuais</div>', unsafe_allow_html=True)

# Inicializa o pré-processador global
preprocessor = TextPreprocessor(
    language=lang_code,
    remove_stopwords=remove_stopwords,
    use_stemming=use_stemming,
    remove_accents=remove_accents
)

# Criação das Abas principais do app
tab1, tab2, tab3 = st.tabs([
    "🔍 Comparação Individual",
    "📊 Comparação Multidocumento",
    "📚 Referencial Teórico"
])

# ---------------------------------------------------------
# ABA 1: COMPARAÇÃO INDIVIDUAL
# ---------------------------------------------------------
with tab1:
    st.markdown("### Comparação entre Dois Documentos")
    
    col_input1, col_input2 = st.columns(2)
    
    with col_input1:
        st.subheader("Documento A")
        input_type_a = st.radio("Entrada do Documento A", ["Digitar Texto", "Upload de Arquivo .txt"], key="input_a_type")
        if input_type_a == "Digitar Texto":
            text_a = st.text_area("Insira o primeiro texto:", placeholder="Escreva ou cole o conteúdo do Documento A aqui...", height=200, key="text_area_a")
        else:
            file_a = st.file_uploader("Selecione o arquivo A:", type=["txt"], key="file_uploader_a")
            text_a = ""
            if file_a:
                text_a = file_a.read().decode('utf-8', errors='replace')
                st.info(f"✓ Arquivo '{file_a.name}' carregado.")
                
    with col_input2:
        st.subheader("Documento B")
        input_type_b = st.radio("Entrada do Documento B", ["Digitar Texto", "Upload de Arquivo .txt"], key="input_b_type")
        if input_type_b == "Digitar Texto":
            text_b = st.text_area("Insira o segundo texto:", placeholder="Escreva ou cole o conteúdo do Documento B aqui...", height=200, key="text_area_b")
        else:
            file_b = st.file_uploader("Selecione o arquivo B:", type=["txt"], key="file_uploader_b")
            text_b = ""
            if file_b:
                text_b = file_b.read().decode('utf-8', errors='replace')
                st.info(f"✓ Arquivo '{file_b.name}' carregado.")

    if st.button("Analisar Similaridade", type="primary", use_container_width=True):
        if not text_a.strip() or not text_b.strip():
            st.warning("⚠️ Por favor, certifique-se de que ambos os documentos possuem conteúdo preenchido.")
        else:
            # Processamento
            tokens_a = preprocessor.preprocess(text_a)
            tokens_b = preprocessor.preprocess(text_b)
            
            # Tokens brutos (sem stemming) para análise de cobertura lexical
            raw_tokens_a = preprocessor.tokenize(text_a)
            raw_tokens_b = preprocessor.tokenize(text_b)
            if preprocessor.remove_stopwords:
                raw_tokens_a = [t for t in raw_tokens_a if t not in preprocessor.stopwords]
                raw_tokens_b = [t for t in raw_tokens_b if t not in preprocessor.stopwords]
            
            str_a = preprocessor.preprocess_as_string(text_a)
            str_b = preprocessor.preprocess_as_string(text_b)
            
            # Vetorização TF-IDF
            try:
                vectorizer = TextVectorizer(use_sublinear_tf=sublinear_tf, ngram_range=ngram_range)
                vectorizer.fit([str_a, str_b])
                vec_a = vectorizer.transform(str_a)
                vec_b = vectorizer.transform(str_b)
                
                # Métricas
                cos_score = SimilarityMetrics.cosine_similarity(vec_a, vec_b)
                jac_score = SimilarityMetrics.jaccard_similarity(tokens_a, tokens_b)
                euc_score = SimilarityMetrics.normalized_euclidean_similarity(vec_a, vec_b)
                
                # Correlação Consolidada
                res = CorrelationEngine.calculate_correlation(cos_score, jac_score, euc_score, weights=user_weights)
                
                # Cobertura Lexical (análise complementar)
                lex_coverage = CorrelationEngine.calculate_lexical_coverage(raw_tokens_a, raw_tokens_b, preprocessor)
                
                # Interpretação contextualizada
                interp = res['interpretation']
                corr = res['correlation_index']
                color = interp['color']
                
                # ---------------------------------------------------------
                # SEÇÃO DE EXIBIÇÃO DE RESULTADOS E CARDS
                # ---------------------------------------------------------
                st.markdown("---")
                st.subheader("🎯 Resultado da Análise")
                
                # Card principal de Índice de Correlação (destaque)
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(22,27,34,0.95) 0%, rgba(13,17,23,0.95) 100%);
                            border: 2px solid {color}; border-radius: 16px; padding: 1.75rem; margin-bottom: 1.5rem;
                            box-shadow: 0 0 25px {color}33;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;">
                        <div>
                            <div style="color: #8b949e; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                                        letter-spacing: 0.08em; margin-bottom: 0.4rem;">Índice de Correlação Consolidado</div>
                            <div style="font-size: 3rem; font-weight: 700; color: {color}; line-height: 1.1;">{corr*100:.1f}%</div>
                            <div style="margin-top: 0.5rem; font-size: 1.05rem; font-weight: 600; color: {color};">
                                {interp['icon']} {interp['label']}
                            </div>
                        </div>
                        <div style="max-width: 55%; min-width: 280px;">
                            <div style="color: #c9d1d9; font-size: 0.9rem; line-height: 1.6; padding: 1rem;
                                        background: rgba(0,0,0,0.3); border-radius: 8px; border-left: 3px solid {color};">
                                {interp['description']}
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Cards de Métricas Individuais
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Similaridade Cosseno</div>
                        <div class="metric-value">{cos_score:.4f}</div>
                        <div class="metric-weight">Peso: {res['weights']['cosine']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Similaridade Jaccard</div>
                        <div class="metric-value">{jac_score:.4f}</div>
                        <div class="metric-weight">Peso: {res['weights']['jaccard']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Similaridade Euclidiana</div>
                        <div class="metric-value">{euc_score:.4f}</div>
                        <div class="metric-weight">Peso: {res['weights']['euclidean']:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ---------------------------------------------------------
                # SEÇÃO: COBERTURA LEXICAL (ANÁLISE COMPLEMENTAR HUMANA)
                # ---------------------------------------------------------
                st.markdown("---")
                st.markdown("### 🧠 Análise de Cobertura Lexical")
                st.caption("Esta análise complementar mostra quantos termos significativos são compartilhados entre os documentos, incluindo correspondências por radical semântico.")
                
                cov = lex_coverage
                cov_pct = cov['coverage_ratio'] * 100
                
                # Cor da barra de cobertura
                if cov_pct >= 50:
                    cov_color = "#26a69a"
                elif cov_pct >= 30:
                    cov_color = "#66bb6a"
                elif cov_pct >= 15:
                    cov_color = "#ffa726"
                else:
                    cov_color = "#b0bec5"
                
                lc1, lc2, lc3, lc4 = st.columns(4)
                with lc1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Correspondências Exatas</div>
                        <div class="metric-value" style="color: #56d364;">{cov['exact_matches']}</div>
                        <div class="metric-weight">palavras idênticas</div>
                    </div>
                    """, unsafe_allow_html=True)
                with lc2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Correspondências por Radical</div>
                        <div class="metric-value" style="color: #e3b341;">{max(cov['stem_matches_doc1'], cov['stem_matches_doc2'])}</div>
                        <div class="metric-weight">mesma raiz semântica</div>
                    </div>
                    """, unsafe_allow_html=True)
                with lc3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Termos Únicos Totais</div>
                        <div class="metric-value" style="color: #c9d1d9;">{cov['total_unique_terms']}</div>
                        <div class="metric-weight">vocabulário combinado</div>
                    </div>
                    """, unsafe_allow_html=True)
                with lc4:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: {cov_color};">
                        <div class="metric-title">Cobertura Lexical</div>
                        <div class="metric-value" style="color: {cov_color};">{cov_pct:.1f}%</div>
                        <div class="metric-weight">termos relacionados</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Barra de progresso visual
                st.markdown(f"""
                <div style="margin-top: 0.75rem; background: rgba(22,27,34,0.8); border-radius: 10px; padding: 0.4rem; 
                            border: 1px solid rgba(48,54,61,0.6);">
                    <div style="display: flex; border-radius: 8px; overflow: hidden; height: 14px;">
                        <div style="width: {(cov['exact_matches'] / max(cov['total_unique_terms'],1)) * 100}%;
                                    background: linear-gradient(90deg, #2ea043, #56d364); transition: width 0.5s ease;" 
                             title="Correspondências exatas"></div>
                        <div style="width: {(max(cov['stem_matches_doc1'], cov['stem_matches_doc2']) / max(cov['total_unique_terms'],1)) * 100}%;
                                    background: linear-gradient(90deg, #d29922, #e3b341); transition: width 0.5s ease;" 
                             title="Correspondências por radical"></div>
                    </div>
                    <div style="display: flex; gap: 1.5rem; margin-top: 0.5rem; padding: 0 0.25rem;">
                        <span style="font-size: 0.75rem; color: #56d364;">■ Exatas</span>
                        <span style="font-size: 0.75rem; color: #e3b341;">■ Radical</span>
                        <span style="font-size: 0.75rem; color: #484f58;">■ Sem correspondência</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Banner explicativo contextual
                if cov_pct >= 20 and corr < 0.35:
                    st.markdown(f"""
                    <div style="margin-top: 1.25rem; padding: 1rem 1.25rem; background: rgba(255,167,38,0.08);
                                border: 1px solid rgba(255,167,38,0.3); border-radius: 10px;">
                        <div style="font-size: 0.9rem; color: #ffa726; font-weight: 600; margin-bottom: 0.4rem;">
                            💡 Observação sobre a Pontuação
                        </div>
                        <div style="font-size: 0.85rem; color: #c9d1d9; line-height: 1.65;">
                            A cobertura lexical ({cov_pct:.0f}%) indica que os textos compartilham vocabulário temático relevante,
                            embora o índice estatístico ({corr*100:.1f}%) esteja abaixo desse valor. Isso é esperado em textos
                            <strong>parafraseados</strong> — que tratam do mesmo assunto utilizando construções e vocabulário
                            diferentes. As métricas TF-IDF e Cosseno medem a similaridade <em>vocabular exata</em>, não a
                            similaridade <em>temática ou semântica</em>. Considere o realce visual abaixo como evidência complementar.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # ---------------------------------------------------------
                # SEÇÃO DE REALCE DE TEXTO COLORIDO (HIGHLIGHT)
                # ---------------------------------------------------------
                st.markdown("")
                st.markdown("### 📝 Alinhamento e Realce Visual das Palavras")
                st.caption("Legenda: ")
                st.markdown("""
                <span class="highlight-exact">Fundo Verde</span> = Correspondência literal exata entre os textos. &nbsp;&nbsp;&nbsp;&nbsp;
                <span class="highlight-stem">Fundo Amarelo</span> = Palavras com mesma raiz semântica (radical). &nbsp;&nbsp;&nbsp;&nbsp;
                Sem Destaque = Palavras únicas e sem relação detectada.
                """, unsafe_allow_html=True)
                st.write("")
                
                html_a, html_b = generate_visual_highlighting(text_a, text_b, preprocessor)
                
                col_text_a, col_text_b = st.columns(2)
                with col_text_a:
                    st.markdown("**Documento A (Visualizador de Correlação):**")
                    st.markdown(f'<div class="text-highlight-box">{html_a}</div>', unsafe_allow_html=True)
                with col_text_b:
                    st.markdown("**Documento B (Visualizador de Correlação):**")
                    st.markdown(f'<div class="text-highlight-box">{html_b}</div>', unsafe_allow_html=True)
                    
            except Exception as ex:
                st.error(f"Ocorreu um erro ao processar os textos: {ex}")

# ---------------------------------------------------------
# ABA 2: COMPARAÇÃO EM LOTE (MULTIDOCUMENTO)
# ---------------------------------------------------------
with tab2:
    st.markdown("### Matriz de Correlação Multidocumento")
    st.write("Faça o upload de múltiplos arquivos `.txt` para cruzar a similaridade de todos com todos.")
    
    uploaded_files = st.file_uploader(
        "Selecione múltiplos arquivos .txt para análise:",
        type=["txt"],
        accept_multiple_files=True,
        key="batch_uploader"
    )
    
    if uploaded_files:
        if len(uploaded_files) < 2:
            st.info("💡 Por favor, faça o upload de pelo menos 2 arquivos para gerar a matriz comparativa.")
        else:
            # Carrega todos os conteúdos textuais
            file_contents = {}
            for f in uploaded_files:
                try:
                    # Tenta ler decodificações adequadas
                    content = f.read().decode('utf-8', errors='replace')
                    if content.strip():
                        file_contents[f.name] = content
                except Exception as ex:
                    st.error(f"Erro ao processar arquivo '{f.name}': {ex}")
                    
            if len(file_contents) < 2:
                st.error("Erro: Não foi possível carregar pelo menos dois arquivos válidos.")
            else:
                st.success(f"✓ {len(file_contents)} arquivos carregados com sucesso. Executando cálculos cruzados...")
                
                # Executa pré-processamento de todos os arquivos
                doc_names = list(file_contents.keys())
                preprocessed_map = {}
                tokens_map = {}
                
                for name, content in file_contents.items():
                    tokens_map[name] = preprocessor.preprocess(content)
                    preprocessed_map[name] = preprocessor.preprocess_as_string(content)
                    
                # Ajusta vetorizador TF-IDF no corpus completo
                try:
                    vectorizer = TextVectorizer(use_sublinear_tf=sublinear_tf, ngram_range=ngram_range)
                    vectorizer.fit(list(preprocessed_map.values()))
                    
                    vectors_map = {
                        name: vectorizer.transform(prep_str)
                        for name, prep_str in preprocessed_map.items()
                    }
                    
                    # Inicialização das matrizes vazias
                    n = len(doc_names)
                    matrix_data = np.zeros((n, n))
                    
                    # Listagem de pares para exibição em tabela
                    pairs_list = []
                    
                    # Loop cruzado para cálculo de métricas
                    for i in range(n):
                        for j in range(n):
                            if i == j:
                                matrix_data[i][j] = 1.0
                            else:
                                name_i = doc_names[i]
                                name_j = doc_names[j]
                                
                                # Métricas
                                cos = SimilarityMetrics.cosine_similarity(vectors_map[name_i], vectors_map[name_j])
                                jac = SimilarityMetrics.jaccard_similarity(tokens_map[name_i], tokens_map[name_j])
                                euc = SimilarityMetrics.normalized_euclidean_similarity(vectors_map[name_i], vectors_map[name_j])
                                
                                # Consolidado
                                res = CorrelationEngine.calculate_correlation(cos, jac, euc, weights=user_weights)
                                corr_index = res['correlation_index']
                                
                                matrix_data[i][j] = corr_index
                                
                                # Salva na lista para ordenação posterior (apenas triangular superior)
                                if i < j:
                                    pairs_list.append({
                                        'Documento 1': name_i,
                                        'Documento 2': name_j,
                                        'Similaridade Cosseno': cos,
                                        'Similaridade Jaccard': jac,
                                        'Similaridade Euclidiana': euc,
                                        'Índice de Correlação': corr_index,
                                        'Interpretação': res['interpretation']['label']
                                    })
                                    
                    # Criação do DataFrame da matriz
                    df_matrix = pd.DataFrame(matrix_data, index=doc_names, columns=doc_names)
                    
                    # ---------------------------------------------------------
                    # PLOTLY HEATMAP INTERATIVO
                    # ---------------------------------------------------------
                    st.markdown("#### 🌡️ Heatmap de Similaridade Consolidada (Correlação %)")
                    
                    fig = px.imshow(
                        df_matrix * 100,
                        text_auto=".1f",
                        aspect="auto",
                        color_continuous_scale=[[0, '#0d1117'], [0.3, '#1f6feb'], [0.7, '#8a2be2'], [1.0, '#00e1d9']],
                        labels=dict(color="Correlação %"),
                        title="Matriz Cruzada de Correlação Textual (em %)"
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#8b949e', family='Inter'),
                        coloraxis_colorbar=dict(title="Correl. %")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # ---------------------------------------------------------
                    # TABELA ORDENADA DE PARES MAIS SIMILARES
                    # ---------------------------------------------------------
                    st.markdown("#### 🏆 Ranking de Pares por Grau de Correlação")
                    df_pairs = pd.DataFrame(pairs_list)
                    df_pairs = df_pairs.sort_values(by='Índice de Correlação', ascending=False)
                    
                    # Formata as colunas numéricas
                    format_dict = {
                        'Similaridade Cosseno': '{:.4f}',
                        'Similaridade Jaccard': '{:.4f}',
                        'Similaridade Euclidiana': '{:.4f}',
                        'Índice de Correlação': '{:.2%}'
                    }
                    
                    st.dataframe(
                        df_pairs.style.format(format_dict).background_gradient(
                            subset=['Índice de Correlação'],
                            cmap='viridis',
                            vmin=0.0,
                            vmax=1.0
                        ),
                        use_container_width=True
                    )
                    
                    # ---------------------------------------------------------
                    # SELETOR PARA COMPARAÇÃO DETALHADA DO LOTE
                    # ---------------------------------------------------------
                    st.markdown("#### 🔍 Comparação Detalhada de um Par Específico do Lote")
                    st.caption("Escolha um par de documentos abaixo para visualizar o alinhamento e realce textual.")
                    
                    col_select1, col_select2 = st.columns(2)
                    with col_select1:
                        doc_a_sel = st.selectbox("Selecione Documento 1:", options=doc_names, index=0)
                    with col_select2:
                        doc_b_sel = st.selectbox("Selecione Documento 2:", options=[d for d in doc_names if d != doc_a_sel], index=0)
                        
                    if doc_a_sel and doc_b_sel:
                        html_da, html_db = generate_visual_highlighting(file_contents[doc_a_sel], file_contents[doc_b_sel], preprocessor)
                        
                        col_view_a, col_view_b = st.columns(2)
                        with col_view_a:
                            st.markdown(f"**Documento: {doc_a_sel}**")
                            st.markdown(f'<div class="text-highlight-box">{html_da}</div>', unsafe_allow_html=True)
                        with col_view_b:
                            st.markdown(f"**Documento: {doc_b_sel}**")
                            st.markdown(f'<div class="text-highlight-box">{html_db}</div>', unsafe_allow_html=True)
                            
                except Exception as ex:
                    st.error(f"Erro ao processar vetorização cruzada do lote: {ex}")

# ---------------------------------------------------------
# ABA 3: REFERENCIAL TEÓRICO
# ---------------------------------------------------------
with tab3:
    st.markdown("### 📚 Fundamentação Teórica e Formulação Matemática")
    st.write("Esta seção descreve a modelagem conceitual utilizada no desenvolvimento deste sistema computacional.")
    
    st.markdown("---")
    
    st.markdown("#### 1. Processamento de Linguagem Natural (PLN)")
    st.write("""
    Antes de submeter os conteúdos textuais aos cálculos estatísticos e vetoriais, os textos passam por um pipeline composto de quatro operações consecutivas:
    """)
    st.markdown("""
    * **Tokenização:** Separação do bloco de texto em termos menores (tokens), isolando pontuações e delimitadores.
    * **Normalização Ortográfica:** Conversão de caracteres maiúsculos para minúsculos e opcionalmente remoção de acentos (diacríticos) via normalização Unicode.
    * **Remoção de Stopwords:** Filtragem de palavras funcionais com alto teor sintático porém vazio valor semântico (como artigos *o, a*, preposições *de, em* e conjunções *e, ou*).
    * **Stemming:** Redução das palavras ao seu radical semântico aproximado. Em português, utiliza-se o algoritmo **RSLP (Representation and Synthesis of Language Part)**, reduzindo flexões verbais e de gênero/número a uma forma canônica comum.
    """)
    
    st.markdown("---")
    
    st.markdown("#### 2. Representação Vetorial TF-IDF")
    st.write("""
    A modelagem vetorial estatística é baseada no algoritmo **TF-IDF (Term Frequency–Inverse Document Frequency)**. A pontuação quantifica a relevância de um termo $t$ em um documento $d$ pertencente a um corpus $D$:
    """)
    
    st.latex(r"TF-IDF(t, d, D) = TF(t, d) \cdot IDF(t, D)")
    
    st.write("Onde a frequência do termo $TF$ com escala sublinear aplicada é formulada como:")
    st.latex(r"TF(t, d) = 1 + \log(f_{t,d}) \quad \text{se } f_{t,d} > 0, \quad \text{caso contrário } 0")
    
    st.write("E a frequência inversa no documento $IDF$ é expressa por:")
    st.latex(r"IDF(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1")
    
    st.write("Cada documento é representado como um vetor $u$ cujos eixos correspondem aos pesos TF-IDF dos termos. Os vetores são normalizados na norma Euclidiana ($L_2$):")
    st.latex(r"\|u\|_2 = \sqrt{\sum_{i=1}^{m} u_i^2} = 1")
    
    st.markdown("---")
    
    st.markdown("#### 3. Métricas de Similaridade")
    
    st.markdown("##### 3.1 Similaridade do Cosseno")
    st.write("Mede o cosseno do ângulo formado pelos vetores unitários $u$ e $v$. Varia de $0$ (ortogonais/sem termos comuns) a $1$ (colineares/termos proporcionais):")
    st.latex(r"S_{cos}(u, v) = \cos(\theta) = \frac{u \cdot v}{\|u\|_2 \|v\|_2} = \sum_{i=1}^{m} u_i v_i")
    
    st.markdown("##### 3.2 Coeficiente de Jaccard")
    st.write("Avalia a similaridade léxica literal a nível de conjuntos de termos pré-processados $A$ e $B$, mensurando a razão entre a interseção e a união:")
    st.latex(r"S_{jac}(A, B) = \frac{|A \cap B|}{|A \cup B|}")
    
    st.markdown("##### 3.3 Similaridade Euclidiana Normalizada (Contribuição Teórica)")
    st.write(r"""
    A distância euclidiana clássica $d(u,v)$ mede a separação geométrica entre os vetores. Como os vetores TF-IDF são normalizados em norma $L_2$, a distância euclidiana máxima na hiperesfera é $\sqrt{2} \approx 1.4142$ (vetores ortogonais). 
    Propomos o cálculo de **Similaridade Euclidiana Normalizada ($S_{euc}$)** contida no intervalo $[0, 1]$:
    """)
    st.latex(r"S_{euc}(u, v) = 1 - \frac{\|u - v\|_2}{\sqrt{2}} = 1 - \frac{\sqrt{\sum_{i=1}^{m} (u_i - v_i)^2}}{\sqrt{2}}")
    
    st.markdown("---")
    
    st.markdown("#### 4. Índice de Correlação Consolidado")
    st.write(r"""
    O Índice de Correlação Consolidado consolida os três indicadores lineares em uma pontuação final interpretável usando pesos normalizados ($\sum w_i = 1$):
    """)
    st.latex(r"\text{Índice Consolidado} = w_{cos} \cdot S_{cos} + w_{jac} \cdot S_{jac} + w_{euc} \cdot S_{euc}")
