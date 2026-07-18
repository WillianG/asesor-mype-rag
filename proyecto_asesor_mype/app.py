import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA

# STREAMING_CHUNK:Configurando el modelo...
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

# STREAMING_CHUNK:Cargando documentos...
def get_vectorstore():
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings)

st.set_page_config(page_title="Asesor Mype IA", page_icon="⚖️")
st.title("⚖️ Asesor Inteligente para Mypes")

try:
    # STREAMING_CHUNK:Inicializando motor RAG...
    db = get_vectorstore()
    retriever = db.as_retriever()
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.chat_input("Ej: ¿Qué es el REMYPE?")
    if query:
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            response = qa.invoke(query)
            st.markdown(response["result"])
except Exception as e:
    st.error(f"Error: Asegúrate de que los PDFs están en la carpeta /data. Detalle: {e}")
```eof