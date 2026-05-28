import os
import sys

class DocumentManager:
    """
    Classe utilitária para leitura e gerenciamento robusto de documentos.
    Suporta múltiplas codificações de caracteres (UTF-8, Latin-1, CP1252)
    para evitar travamentos comuns em sistemas Windows/Linux.
    """
    
    @staticmethod
    def read_file(file_path):
        """
        Lê o conteúdo textual de um arquivo de forma altamente resiliente.
        Tenta codificações comuns em sequência caso ocorra erro.
        
        :param file_path: Caminho completo para o arquivo.
        :return: String com o conteúdo textual do arquivo.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"O arquivo '{file_path}' não foi encontrado.")
            
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
                
        # Se todas falharem, lê com 'replace' para não derrubar o sistema
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
        except Exception as e:
            print(f"[Erro] Falha catastrófica ao ler {file_path}: {e}", file=sys.stderr)
            return ""

    @classmethod
    def load_documents_from_directory(cls, directory_path):
        """
        Carrega todos os arquivos de extensão .txt presentes em um diretório.
        
        :param directory_path: Caminho completo do diretório.
        :return: Dicionário mapeando {nome_do_arquivo: conteudo_textual}.
        """
        documents = {}
        if not os.path.isdir(directory_path):
            print(f"[Aviso] O diretório '{directory_path}' não é válido ou não existe.", file=sys.stderr)
            return documents
            
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                try:
                    content = cls.read_file(file_path)
                    # Registra o documento apenas se não estiver vazio
                    if content.strip():
                        documents[filename] = content
                except Exception as e:
                    print(f"[Erro] Falha ao carregar '{filename}': {e}", file=sys.stderr)
                    
        return documents
