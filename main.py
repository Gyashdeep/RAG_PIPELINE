import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Page Config
st.set_page_config(page_title="Enterprise AI Data Factory", layout="wide")
st.title("🤖 Enterprise AI Data Factory")

# Sidebar for API Key
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# File Uploader
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    # Save temporary file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # LLM Setup
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, groq_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever())
    
    # Query Interface
    query = st.text_input("Ask a question about the PDF:")
    if query:
        with st.spinner("Processing..."):
            response = qa_chain.invoke(query)
            st.markdown("### Answer")
            st.write(response['result'])
