import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Asesor Mype IA", page_icon="⚖️")

def main():
    st.title("⚖️ Asesor Inteligente para Mypes")
    
    # Verificación de datos
    if not os.path.exists("./data") or not os.listdir("./data"):
        st.warning("Por favor, sube archivos PDF a la carpeta './data'.")
        return

    # Usamos embeddings de Google (no requieren instalación de librerías extra)
    @st.cache_resource
    def load_rag_engine():
        loader = PyPDFDirectoryLoader("./data")
        docs = loader.load()
        
        # Embeddings nativos de Google
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        db = Chroma.from_documents(docs, embeddings)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=db.as_retriever()
        )

    try:
        qa = load_rag_engine()
        query = st.chat_input("¿Qué beneficios tiene el REMYPE?")
        
        if query:
            with st.chat_message("user"): st.markdown(query)
            with st.chat_message("assistant"):
                response = qa.invoke(query)
                st.markdown(response["result"])
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
```eof
