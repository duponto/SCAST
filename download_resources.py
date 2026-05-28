import nltk
import sys

def download_resources():
    print("====================================================")
    print("Iniciando o download de recursos necessários do NLTK")
    print("====================================================")
    
    resources = ['punkt', 'stopwords', 'rslp']
    
    for resource in resources:
        try:
            print(f"Verificando / Baixando: {resource}...")
            nltk.download(resource, quiet=False)
            print(f"✓ Recurso '{resource}' carregado com sucesso.")
        except Exception as e:
            print(f"✗ Erro ao baixar recurso '{resource}': {e}", file=sys.stderr)
            
    print("====================================================")
    print("Download de recursos do NLTK finalizado.")
    print("====================================================")

if __name__ == "__main__":
    download_resources()
