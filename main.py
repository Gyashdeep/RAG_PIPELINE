import streamlit as st
import os
import tempfile

# Standard imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Enterprise AI Data Factory", layout="wide")
st.title("🤖 Enterprise AI Data Factory")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and api_key:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        with st.spinner("Processing..."):
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            retriever = vector_db.as_retriever()
            
            llm = ChatGroq(model="llama-3-70b-8192", temperature=0, groq_api_key=api_key)
            
            prompt = ChatPromptTemplate.from_template("""Answer the question based on context: {context} \n Question: {input}""")
            
            combine_docs_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        query = st.text_input("Ask a question:")
        if query:
            res = retrieval_chain.invoke({"input": query})
            st.write(res['answer'])
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
