"""
Módulo de inicialização e cache de recursos NLTK para Streamlit.
Garante que recursos do NLTK sejam baixados automaticamente na primeira execução.
"""

import nltk
import os
import sys
from pathlib import Path

def ensure_nltk_resources():
    """
    Verifica e baixa recursos NLTK necessários para o funcionamento correto do sistema.
    Esta função é chamada uma única vez durante a inicialização do Streamlit.
    """
    resources_needed = ['punkt', 'stopwords', 'rslp']
    resources_missing = []
    
    for resource in resources_needed:
        try:
            # Tenta acessar o recurso
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'rslp':
                nltk.data.find('tokenizers/rslp')
        except LookupError:
            resources_missing.append(resource)
    
    # Se houver recursos faltando, baixa-os
    if resources_missing:
        print(f"[INFO] Baixando recursos NLTK faltando: {', '.join(resources_missing)}", 
              file=sys.stderr)
        for resource in resources_missing:
            try:
                nltk.download(resource, quiet=True)
                print(f"[OK] Recurso '{resource}' baixado com sucesso.", file=sys.stderr)
            except Exception as e:
                print(f"[ERRO] Falha ao baixar recurso '{resource}': {e}", file=sys.stderr)
    else:
        print("[OK] Todos os recursos NLTK já estão disponíveis.", file=sys.stderr)

# Executar na importação do módulo
if __name__ != "__main__":
    try:
        ensure_nltk_resources()
    except Exception as e:
        print(f"[AVISO] Erro durante inicialização NLTK: {e}", file=sys.stderr)
