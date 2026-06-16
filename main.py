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

# 1. Cached Embedding Model (Loads once)
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 2. Cached Vector Store (Only re-processes if the file changes)
@st.cache_resource(show_spinner=False)
def get_vector_db(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())

# Sidebar Configuration
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# 3. Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file and api_key:
    # Save file temporarily to process
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        vector_db = get_vector_db(tmp_path)
        retriever = vector_db.as_retriever()
        llm = ChatGroq(model="llama-3-70b-8192", temperature=0, groq_api_key=api_key)
        
        prompt = ChatPromptTemplate.from_template("""Answer based on context: {context} \n Question: {input}""")
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Display Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # User Query
        if query := st.chat_input("Ask a question about your PDF:"):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.spinner("Thinking..."):
                res = retrieval_chain.invoke({"input": query})
                answer = res['answer']
                
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

    except Exception as e:
        st.error(f"Execution Error: {e}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
