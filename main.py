import streamlit as st
import os
import tempfile
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
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            retriever = vector_db.as_retriever()
            
            llm = ChatGroq(model="llama-3-70b-8192", temperature=0, groq_api_key=api_key)
            
            # New LCEL Chain setup
            prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}""")
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Generating answer..."):
                response = retrieval_chain.invoke({"input": query})
                st.markdown("### Answer")
                st.write(response['answer'])
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
else:
    st.info("Please provide your Groq API Key and upload a PDF.")
