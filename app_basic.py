"""
Assistente RAG de Políticas Internas (RH)
- Streamlit (interface)
- LangChain com LCEL (orquestração)
- InMemoryVectorStore (banco vetorial em memória, do langchain-core)
- HuggingFace (embeddings locais, grátis) + Google Gemini (LLM)
"""
# Sistema RAG
# Embeddings

import io
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser


# Carrega a API key: no Streamlit Cloud vem de st.secrets; localmente, do .env
load_dotenv()
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    # Nenhum arquivo de secrets configurado (rodando localmente com .env) — segue o baile
    pass

PDF_PADRAO = "politica_rh.pdf"

# logica do contexto do RAG
def carregar_pdf(fonte): # ler o pdf (caminho ou arquivo em memória) e extrair o texto de cada página
    leitor = PdfReader(fonte)
    documentos = []
    for numero_pagina, pagina in enumerate(leitor.pages):
        texto = pagina.extract_text() or ""
        if texto.strip():
            # metadata "page" (índice 0) mantém compatibilidade com a UI de fontes
            documentos.append(
                Document(page_content=texto, metadata={"page": numero_pagina})
            )
    return documentos

def separar_em_blocos(texto_pdf):
    separador = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200)
    lista_blocos = separador.split_documents(texto_pdf)
    return lista_blocos

# Embedding
def criar_banco_vetores(lista_blocos):
    ferramenta_embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    banco_vetores = InMemoryVectorStore.from_documents(lista_blocos, ferramenta_embedding)
    return banco_vetores

# @st.cache_resource: roda uma única vez por PDF e reaproveita o resultado entre
# execuções/reloads do Streamlit, evitando reprocessar o mesmo documento e
# recriar os embeddings a cada pergunta. A chave do cache são os bytes do PDF,
# então enviar um arquivo diferente reprocessa automaticamente.
@st.cache_resource(show_spinner="Processando o PDF e gerando embeddings...")
def preparar_banco_vetores(pdf_bytes):
    texto_pdf = carregar_pdf(io.BytesIO(pdf_bytes))
    lista_blocos = separar_em_blocos(texto_pdf)
    return criar_banco_vetores(lista_blocos)

# logica do agente de IA
def criar_chain_agente(banco_vetores):

    prompt_template = ChatPromptTemplate.from_template(
        """Você é um assistente de RH que responde perguntas sobre políticas internas da empresa.
    Use APENAS as informações do contexto abaixo para responder.
    Se não encontrar a resposta, diga claramente que não sabe responder.
    Responda em português do Brasil, de forma clara e objetiva.

    Contexto: {context}

    A pergunta: {question}

    Resposta:""")

    buscador_contexto = banco_vetores.as_retriever()

    # alias "latest": aponta sempre para o Gemini Flash atual, evitando quebra
    # quando uma versão específica é descontinuada pela Google
    llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

    # junta o conteúdo dos blocos recuperados em um único texto de contexto
    def formatar_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # gera a resposta a partir do dict {"docs": [...], "question": "..."}
    gerar_resposta = (

        RunnablePassthrough.assign(context=lambda x: formatar_docs(x["docs"]))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # a chain recupera os documentos, mantém a pergunta e adiciona a resposta,
    # devolvendo tudo para que possamos exibir as fontes na interface
    chain = (
        RunnableParallel(docs=buscador_contexto, question=RunnablePassthrough())
        .assign(answer=gerar_resposta)
    )

    return chain




# interface

st.title("Assistente RAG — Políticas Internas")

# Permite enviar qualquer PDF; sem upload, usa o documento de exemplo padrão
with st.sidebar:
    st.header("📁 Documento")
    arquivo_enviado = st.file_uploader("Envie um PDF (opcional)", type="pdf")
    if arquivo_enviado is not None:
        pdf_bytes = arquivo_enviado.getvalue()
        st.success(f"Usando: {arquivo_enviado.name}")
    else:
        with open(PDF_PADRAO, "rb") as f:
            pdf_bytes = f.read()
        st.info(f"Usando o documento de exemplo: {PDF_PADRAO}")

# carregar chain e vector store (o banco é cacheado por PDF)
banco_vetores = preparar_banco_vetores(pdf_bytes)
chain = criar_chain_agente(banco_vetores)


# inicia a lista de mensagens
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# exibe as mensagens na tela
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Pergunte sobre as políticas da empresa...")
if prompt :
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            resultado = chain.invoke(prompt)
            answer = resultado["answer"]
            docs_fonte = resultado["docs"]
            st.write(answer)

            # exibe os trechos do PDF que embasaram a resposta
            with st.expander(f"📄 Fontes ({len(docs_fonte)} trechos)"):
                for i, doc in enumerate(docs_fonte, start=1):
                    pagina = doc.metadata.get("page")
                    rotulo = f"**Trecho {i}**"
                    if pagina is not None:
                        rotulo += f" — página {pagina + 1}"
                    st.markdown(rotulo)
                    st.caption(doc.page_content)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
