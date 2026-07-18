import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asesor Mype AI", page_icon="⚖️")
st.title("⚖️ Asesor Inteligente para Mypes")
st.markdown("Asistente basado en normativa SUNAT/SUNAFIL (Demo Gratuita)")

# Configuración de servicios
def get_llm():
    # Usamos la API Key desde los secrets de Streamlit
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.secrets["GOOGLE_API_KEY"])

def get_vectorstore():
    # Carga de documentos desde la carpeta /data
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    # Embeddings locales (no requieren pago)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(docs, embeddings)

try:
    if os.path.exists("./data") and os.listdir("./data"):
        db = get_vectorstore()
        retriever = db.as_retriever()
        llm = get_llm()
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        query = st.chat_input("Ej: ¿Qué beneficios tiene una microempresa?")
        
        if query:
            with st.chat_message("user"): 
                st.markdown(query)
            with st.chat_message("assistant"):
                response = qa.invoke(query)
                st.markdown(response["result"])
                st.info("Disclaimer: Este bot es informativo. No reemplaza asesoría profesional.")
    else:
        st.warning("Por favor, asegúrate de tener archivos PDF en la carpeta './data'.")
except Exception as e:
    st.error(f"Error técnico: {str(e)}")
