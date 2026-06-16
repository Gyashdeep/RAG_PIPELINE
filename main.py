import streamlit as st
import os

# Set layout immediately
st.set_page_config(page_title="Enterprise AI Data Factory", layout="wide")

# Add a try-except block around imports to see exactly what is missing
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
except ImportError as e:
    st.error(f"Missing dependency error: {e}")
    st.stop()

st.title("🤖 Enterprise AI Data Factory")

# Sidebar
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    try:
        # Save file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load and process
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        llm = ChatGroq(model="llama-3-70b-8192", temperature=0, groq_api_key=api_key)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
        
        query = st.text_input("Ask a question about the PDF:")
        if query:
            response = qa_chain.invoke(query)
            st.markdown("### Answer")
            st.write(response['result'])
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
