import streamlit as st
import os
# Importamos desde los sub-paquetes directamente para mayor estabilidad
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Asesor Mype IA", page_icon="⚖️")

def main():
    st.title("⚖️ Asesor Inteligente para Mypes")
    
    if not os.path.exists("./data") or not os.listdir("./data"):
        st.warning("Asegúrate de que la carpeta 'data' contenga archivos PDF.")
        return

    @st.cache_resource
    def load_rag_engine():
        loader = PyPDFDirectoryLoader("./data")
        docs = loader.load()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        db = Chroma.from_documents(docs, embeddings)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            google_api_key=st.secrets["GOOGLE_API_KEY"]
        )
        
        # Usamos la cadena moderna recomendada por LangChain
        prompt = ChatPromptTemplate.from_template("""
        Responde basándote en el contexto: {context}
        Pregunta: {input}
        """)
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        return create_retrieval_chain(db.as_retriever(), combine_docs_chain)

    try:
        rag_chain = load_rag_engine()
        query = st.chat_input("¿Qué beneficios tiene el REMYPE?")
        
        if query:
            with st.chat_message("user"): st.markdown(query)
            with st.chat_message("assistant"):
                response = rag_chain.invoke({"input": query})
                st.markdown(response["answer"])
    except Exception as e:
        st.error(f"Error técnico: {str(e)}")

if __name__ == "__main__":
    main()
