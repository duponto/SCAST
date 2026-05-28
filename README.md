# SCAST - Sistema de Comparação e Análise de Similaridade Textual

O **SCAST** é um sistema computacional de alto desempenho desenvolvido do zero em Python, voltado para a comparação de documentos, análise de similaridade e cálculo de índices de correlação textual. O sistema combina Processamento de Linguagem Natural (PLN) avançado, modelagem estatística vetorial e fusão de métricas matemáticas lineares em um painel web interativo e moderno.

---

## 🌟 Principais Recursos e Diferenciais Teóricos

1. **Fusão Multimétrica Ponderada:** 
   O sistema calcula a similaridade através de três perspectivas matemáticas distintas:
   * **Similaridade de Cosseno (Ângulo Vetorial):** Mede a orientação dos vetores de termos TF-IDF.
   * **Coeficiente de Jaccard (Léxica/Conjuntos):** Mede a sobreposição direta de palavras únicas.
   * **Similaridade Euclidiana Normalizada (Geométrica):** Proposta teórica inovadora para converter a distância Euclidiana dos vetores de norma $L_2$ em uma similaridade restrita ao intervalo $[0, 1]$ pela fórmula:
     $$S_{euc} = 1 - \frac{d(u, v)}{\sqrt{2}}$$
   * O **Índice de Correlação Consolidado** final é a combinação ponderada destas métricas com pesos ajustáveis em tempo real.

2. **Visualização de Alinhamento e Realce Altamente Assertivo:**
   A interface web alinha os textos lado a lado e destaca graficamente as palavras de acordo com sua semelhança:
   * <span style="background-color: rgba(46, 160, 67, 0.2); border: 1px solid rgba(46, 160, 67, 0.4); border-radius: 4px; padding: 2px 4px; color: #56d364;">Fundo Verde</span>: Palavras idênticas literalmente.
   * <span style="background-color: rgba(210, 153, 34, 0.2); border: 1px solid rgba(210, 153, 34, 0.4); border-radius: 4px; padding: 2px 4px; color: #e3b341;">Fundo Amarelo</span>: Palavras com radicais semânticos idênticos (Stemming).
   * **Supressão de Stopwords:** Palavras funcionais comuns (*artigos, preposições, pronomes*) são automaticamente excluídas do realce visual para eliminar poluição visual e focar estritamente no conteúdo intelectual relevante.

3. **Análise de Lote com Heatmaps Interativos:**
   Carregue dezenas de arquivos `.txt` de uma só vez para gerar cruzamentos estatísticos automáticos com gráficos de calor (Heatmaps) em 2D interativos gerados pelo Plotly e rankings de similaridade ordenáveis.

---

## 📂 Arquitetura do Projeto

O código é estruturado de maneira modular e extensível:

```text
py-textual-similarities/
│
├── .venv/                      # Ambiente Virtual Python (Isolamento de Pacotes)
├── requirements.txt            # Dependências (Streamlit, Scikit-Learn, Plotly, etc.)
├── download_resources.py       # Script de download utilitário do NLTK (Corpora, Stemmers)
├── test_engine.py              # Suíte de Testes Automatizados (Consistência Matemática)
├── cli.py                      # Módulo de Terminal (CLI)
├── app.py                      # Dashboard Web Interativo (Streamlit)
│
├── engine/                     # Motor Central de Similaridade
│   ├── __init__.py             # Exposição de interfaces limpas
│   ├── preprocessor.py         # Tokenização, Stopwords, Stemming (RSLP PT/Snowball EN)
│   ├── vectorizer.py           # Modelagem TF-IDF com suporte a N-grams
│   ├── metrics.py              # Cosseno, Jaccard e Euclidiana Normalizada
│   ├── correlation.py          # Fusão de pesos e análises semânticas qualitativas
│   └── document_manager.py     # Leitor de arquivos resiliente a encondings (Latin-1/CP1252/UTF-8)
│
└── sample_docs/                # Documentos de Exemplo para Teste Rápido
    ├── documento_a.txt         # Artigo original sobre IA
    ├── documento_b.txt         # Paráfrase de IA (Alta similaridade semântica)
    └── documento_c.txt         # Artigo sobre futebol (Similaridade nula)
```

---

## 🚀 Como Configurar e Rodar o Projeto (Passo a Passo)

### 📌 Pré-requisitos
- **Python 3.10+** (ou superior) instalado no seu sistema
- **pip** (gerenciador de pacotes Python - geralmente vem junto com o Python)
- **Git** (opcional, apenas se quiser clonar de um repositório)

#### ✅ Verificar Instalação do Python
Abra seu terminal e execute:
```bash
python --version
```
Você deverá ver algo como `Python 3.12.10` (ou versão superior).

---

### 1️⃣ Abrir a Pasta do Projeto
Abra seu terminal (PowerShell, Prompt de Comando CMD ou terminal do VS Code) e navegue até a pasta raiz do projeto:

**Windows (PowerShell):**
```powershell
cd D:\dev\py-textual-similarities
```

**Windows (CMD):**
```cmd
cd D:\dev\py-textual-similarities
```

**Linux/macOS:**
```bash
cd /path/to/py-textual-similarities
```

---

### 2️⃣ Criar e Ativar o Ambiente Virtual (`.venv`)
O ambiente virtual isola as dependências deste projeto, evitando conflitos com outras instalações Python no seu sistema.

#### 2.1 - Criar o Ambiente Virtual
```bash
python -m venv .venv
```

#### 2.2 - Ativar o Ambiente Virtual
Agora você **precisa ativar** a `.venv` para que todos os comandos usem os pacotes isolados do projeto:

* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
  
  ⚠️ *Se receber erro de permissão, execute:*
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
  *Depois tente ativar novamente.*

* **Windows (Prompt de Comando CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```

* **Linux / macOS (Terminal):**
  ```bash
  source .venv/bin/activate
  ```

> 💡 **Dica:** Após a ativação bem-sucedida, você verá o prefixo `(.venv)` no início da linha de comando. Exemplo:
> ```
> (.venv) D:\dev\py-textual-similarities>
> ```

---

### 3️⃣ Instalar as Dependências do Projeto
Com a `.venv` ativa, instale todos os pacotes necessários listados em `requirements.txt`:

```bash
pip install -r requirements.txt
```

Isso instalará:
- **streamlit** (Dashboard web interativo)
- **nltk** (Processamento de linguagem natural)
- **scikit-learn** (Machine learning e vetorização TF-IDF)
- **pandas** (Manipulação de dados)
- **numpy** (Operações numéricas)
- **plotly** (Gráficos interativos)
- **matplotlib** (Suporte a estilos de dados)
- **pytest** (Framework de testes)

⏱️ *A instalação pode levar 2-5 minutos, dependendo da sua conexão de internet.*

---

### 4️⃣ Baixar Recursos Necessários do NLTK
O NLTK requer recursos adicionais (tokenizadores, listas de stopwords, stemmers) que devem ser baixados uma única vez:

```bash
python download_resources.py
```

Este script baixará automaticamente:
- ✅ **punkt** - Tokenizador (divisor de textos em sentenças e palavras)
- ✅ **stopwords** - Listas de palavras vazias (português e inglês)
- ✅ **rslp** - Stemmer RSLP para redução de palavras em português

> 💡 **Nota:** Se os recursos já estiverem baixados, o script apenas confirmará isso e não fará downloads redundantes.

---

### 5️⃣ Rodar a Interface Web (Dashboard Streamlit) ⭐
Esta é a forma **recomendada** de usar o sistema com a interface visual completa:

```bash
streamlit run app.py
```

**Resultado esperado:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.31.17:8501
```

O navegador **abrirá automaticamente** a aplicação em `http://localhost:8501`.

Se não abrir automaticamente, acesse manualmente o link acima.

#### Funcionalidades Disponíveis:
- 🔍 **Aba 1: Comparação Individual** - Compare dois documentos com análise detalhada e realce visual
- 📊 **Aba 2: Comparação Multidocumento** - Analise múltiplos arquivos com heatmaps interativos e rankings
- 📚 **Aba 3: Referencial Teórico** - Consulte as fórmulas matemáticas e explicações detalhadas

---

### 6️⃣ (Opcional) Rodar a Interface de Terminal (CLI)
Para comparações rápidas sem abrir o navegador, use a interface de linha de comando:

#### Exemplo Básico:
```bash
python cli.py --file1 sample_docs/documento_a.txt --file2 sample_docs/documento_b.txt
```

#### Ver Todas as Opções:
```bash
python cli.py --help
```

#### Exemplo Avançado (com ajustes de pesos e idioma):
```bash
python cli.py --file1 doc1.txt --file2 doc2.txt --lang portuguese --w-cos 0.6 --w-jac 0.2 --w-euc 0.2
```

**Opções disponíveis:**
- `--file1 CAMINHO` - Caminho do primeiro arquivo (.txt)
- `--file2 CAMINHO` - Caminho do segundo arquivo (.txt)
- `--text1 TEXTO` - Texto direto em vez de arquivo
- `--text2 TEXTO` - Texto direto em vez de arquivo
- `--lang [portuguese|english]` - Idioma (padrão: portuguese)
- `--no-stemming` - Desativa redução de palavras aos radicais
- `--no-stopwords` - Desativa remoção de palavras vazias
- `--w-cos PESO` - Peso da Similaridade de Cosseno (0.0-1.0, padrão: 0.5)
- `--w-jac PESO` - Peso da Similaridade de Jaccard (0.0-1.0, padrão: 0.3)
- `--w-euc PESO` - Peso da Similaridade Euclidiana (0.0-1.0, padrão: 0.2)

---

### 7️⃣ (Opcional) Rodar a Suíte de Testes Automatizados
Para validar a integridade matemática e funcional do sistema:

```bash
python test_engine.py
```

Isso executará todos os testes e mostrará um relatório de cobertura de funcionalidades.

---

## 🔧 Troubleshooting (Solução de Problemas)

### ❌ Erro: "Python não foi encontrado"
**Solução:** Certifique-se de que Python está instalado e adicionado ao PATH. Verifique com:
```bash
python --version
```
Se não funcionar, instale Python do site oficial: https://www.python.org/downloads/

---

### ❌ Erro: "ModuleNotFoundError: No module named 'streamlit'"
**Solução:** A `.venv` não está ativa ou os pacotes não foram instalados. Execute:
```bash
# Ativar ambiente virtual (escolha conforme seu SO):
.venv\Scripts\Activate.ps1           # Windows PowerShell
.venv\Scripts\activate.bat           # Windows CMD
source .venv/bin/activate            # Linux/macOS

# Depois instalar dependências:
pip install -r requirements.txt
```

---

### ❌ Erro: "matplotlib failed" ou "Styler.background_gradient"
**Solução:** Matplotlib não foi instalado. Execute:
```bash
pip install matplotlib>=3.7.0
```

---

### ❌ Erro: "NLTK resources not found" (punkt, stopwords, rslp)
**Solução:** Baixe os recursos com:
```bash
python download_resources.py
```

---

### ⚠️ Porta 8501 já está em uso
**Solução:** A porta padrão do Streamlit já está ocupada. Execute com outra porta:
```bash
streamlit run app.py --server.port 8502
```
Ou finalize qualquer processo anterior rodando em 8501.

---

### 💡 A aplicação abriu mas está lenta
**Solução:** Isso é normal na primeira execução. Streamlit faz cache. Recarregue a página ou reinicie a aplicação. Nas próximas vezes será mais rápido.

---

## 📊 Testando com Documentos de Exemplo
O projeto inclui 3 documentos de exemplo em `sample_docs/` para testes rápidos:

- **documento_a.txt** - Artigo sobre Inteligência Artificial
- **documento_b.txt** - Paráfrase do documento A (Alta similaridade esperada ~0.7-0.9)
- **documento_c.txt** - Artigo sobre Futebol (Similaridade nula com A e B)

### Via Interface Web:
1. Abra a aba "Comparação Individual" ou "Comparação Multidocumento"
2. Faça upload dos arquivos em `sample_docs/`
3. Clique em "Analisar Similaridade"

### Via Terminal (CLI):
```bash
python cli.py --file1 sample_docs/documento_a.txt --file2 sample_docs/documento_b.txt
```

---

## 🛠️ Tecnologias Utilizadas
* **Python 3.12**
* **Streamlit** (Painel Web e UX Visual)
* **Scikit-Learn** (Vetorização TF-IDF e Distância Euclidiana)
* **NLTK** (Tokenização, Stemming RSLP, Stopwords)
* **Plotly** (Matrizes de calor interativas)
* **Pandas & NumPy** (Manipulação de dados estatísticos estruturados)

---

## 📋 Comandos Rápidos (Resumo)

Depois de executar os passos 1-4 acima, use estes comandos:

| Comando | Função |
|---------|--------|
| `streamlit run app.py` | Iniciar dashboard web em http://localhost:8501 |
| `python cli.py --help` | Ver opções de linha de comando |
| `python cli.py --file1 A.txt --file2 B.txt` | Comparar dois arquivos via terminal |
| `python test_engine.py` | Rodar testes automatizados |
| `python download_resources.py` | Atualizar/baixar recursos NLTK |
| `.venv\Scripts\deactivate` | Desativar ambiente virtual |

---

## 📁 Estrutura de Arquivos Importante

```
.venv/                    ← Não editar! Ambiente isolado com todas as dependências
requirements.txt          ← Lista de pacotes Python (não modificar sem razão)
download_resources.py     ← Script para baixar recursos NLTK (execute uma vez)
app.py                    ← Interface web Streamlit (PRINCIPAL)
cli.py                    ← Interface terminal (opcional)
test_engine.py           ← Testes automatizados (opcional)
engine/                  ← Módulos de processamento (núcleo do sistema)
sample_docs/             ← Documentos de exemplo para testes
```

---

## 🌐 Referências e Links Úteis

### Documentação Oficial:
- **Streamlit Docs:** https://docs.streamlit.io/
- **NLTK Docs:** https://www.nltk.org/
- **Scikit-Learn Docs:** https://scikit-learn.org/
- **Python Official:** https://www.python.org/doc/

### Tutoriais Relacionados:
- Como usar ambientes virtuais Python: https://docs.python.org/3/tutorial/venv.html
- Introdução a TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Similaridade de Cosseno: https://en.wikipedia.org/wiki/Cosine_similarity

---

## ✨ Dicas e Boas Práticas

### ✅ Sempre Use o Ambiente Virtual
- Nunca instale pacotes com `pip install` sem ter a `.venv` ativa
- Isso garante que o projeto é isolado e reproduzível em qualquer máquina

### ✅ Atualize os Requisitos se Adicionar Pacotes
Se você instalar novos pacotes:
```bash
pip freeze > requirements.txt
```
Isso atualiza o arquivo de dependências.

### ✅ Use Textos Longos para Melhor Precisão
- Textos muito curtos (1-2 palavras) tendem a ter baixa similaridade mesmo se semelhantes
- Melhor desempenho com documentos de 50+ palavras

### ✅ Personalize os Pesos Conforme Necessário
Os pesos padrão (Cosseno: 0.5, Jaccard: 0.3, Euclidiana: 0.2) são balanceados
mas você pode ajustá-los via interface ou CLI conforme sua necessidade.

### ✅ Teste a CLI Antes de Usar o Streamlit
```bash
python cli.py --file1 sample_docs/documento_a.txt --file2 sample_docs/documento_b.txt
```
Se isso funcionar, a interface web também funcionará.

---

## 🐛 Relatórios de Problemas

Se encontrar algum erro após seguir todos os passos:

1. **Verifique se a `.venv` está ativa** (deve estar escrito `(.venv)` no terminal)
2. **Reinstale as dependências:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```
3. **Verifique a versão do Python:**
   ```bash
   python --version
   ```
   Deve ser 3.10 ou superior.

4. **Limpe cache do Streamlit** (se houver problemas com a interface web):
   ```bash
   streamlit cache clear
   ```

---

## 📝 Licença e Uso

Este projeto foi desenvolvido como um sistema de análise de similaridade textual baseado em PLN avançado. Sinta-se livre para adaptar e estender conforme necessário.

---

## 🎯 Próximos Passos Após Instalação

1. ✅ Abra a interface web: `streamlit run app.py`
2. ✅ Teste com os documentos de exemplo em `sample_docs/`
3. ✅ Experimente ajustar os pesos e opções de processamento
4. ✅ Faça upload de seus próprios documentos `.txt`
5. ✅ Explore a aba "Referencial Teórico" para entender as fórmulas matemáticas

---

**Pronto para começar! 🚀 Qualquer dúvida, verifique a seção de Troubleshooting acima ou consulte a documentação oficial das bibliotecas.**
