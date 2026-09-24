# Assistente RAG — Políticas Internas de RH

Um assistente conversacional que responde perguntas sobre as políticas internas de RH de uma empresa. Ele usa **RAG (Retrieval-Augmented Generation)**: as respostas são geradas por um LLM, mas apenas com base no conteúdo de um PDF de políticas — evitando "invenções" fora do documento.

## Como funciona

1. **Carrega** o PDF `politica_rh.pdf` e extrai o texto.
2. **Divide** o texto em blocos (chunks) de 1000 caracteres com sobreposição de 200.
3. **Gera embeddings** locais de cada bloco com um modelo HuggingFace multilíngue.
4. **Indexa** os embeddings em um banco vetorial FAISS.
5. Na pergunta do usuário, **busca** os blocos mais relevantes e os envia como contexto para o **Google Gemini**, que redige a resposta em português.

## Tecnologias

| Camada | Ferramenta |
|--------|-----------|
| Interface | [Streamlit](https://streamlit.io/) |
| Orquestração | [LangChain](https://www.langchain.com/) (LCEL) |
| Banco vetorial | [FAISS](https://github.com/facebookresearch/faiss) |
| Embeddings | HuggingFace (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) — locais e gratuitos |
| LLM | Google Gemini (`langchain-google-genai`) |

## Pré-requisitos

- Python 3.9+
- Uma chave de API do Google Gemini ([Google AI Studio](https://aistudio.google.com/app/apikey))
- O arquivo `politica_rh.pdf` na raiz do projeto

## Instalação

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>
cd Agente_rh

# 2. (Recomendado) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate   # no Windows: .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt
```

## Configuração

Crie um arquivo `.env` na raiz do projeto com sua chave de API:

```env
GOOGLE_API_KEY=sua_chave_aqui
```

## Como executar

```bash
streamlit run app_basic.py
```

O aplicativo abre no navegador (por padrão em `http://localhost:8501`). Digite suas perguntas no campo de chat e o assistente responde com base nas políticas de RH.

## Estrutura do projeto

```
Agente_rh/
├── app_basic.py        # Aplicação principal (RAG + interface Streamlit)
├── politica_rh.pdf     # Documento-fonte com as políticas de RH
├── requirements.txt    # Dependências do projeto
└── README.md
```

## Observações

- Os embeddings são gerados **localmente** e de graça; apenas o LLM (Gemini) consome sua cota de API.
- Para usar outro documento, basta substituir o `politica_rh.pdf`.
- O índice FAISS é reconstruído a cada inicialização do app.
