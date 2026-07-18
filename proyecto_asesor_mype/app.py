import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asesor Mype AI", page_icon="⚖️")

# Carga de documentos desde la carpeta /data
@st.cache_resource
def get_vectorstore():
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    # Usamos la API Key desde los secretos de Streamlit
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    return Chroma.from_documents(docs, embeddings)

st.title("⚖️ Asesor Inteligente para Mypes")
st.markdown("Asistente experto en normativa SUNAT/SUNAFIL.")

try:
    db = get_vectorstore()
    retriever = db.as_retriever()
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=st.secrets["OPENAI_API_KEY"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    query = st.chat_input("Ej: ¿Qué beneficios tiene una microempresa?")
    if query:
        with st.chat_message("user"): st.markdown(query)
        with st.chat_message("assistant"):
            response = qa.invoke(query)
            st.markdown(response["result"])
            st.info("Disclaimer: Este bot es informativo. No reemplaza asesoría profesional.")
except Exception as e:
    st.error("Error al cargar los documentos. Asegúrate de tener la API Key en los secretos.")