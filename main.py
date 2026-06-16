import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# 1. Load and Chunk PDF
loader = PyPDFLoader("your_document.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# 2. Embedding and Vector Storage
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(chunks, embeddings)

# 3. Setup Groq with the GPT-OSS-120B model
# Ensure your GROQ_API_KEY is set in your environment
llm = ChatGroq(
    model="openai/gpt-oss-120b", 
    temperature=0,
    groq_api_key=os.environ["GROQ_API_KEY"]
)

# 4. Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever()
)

# 5. Querying with a prompt for polished output
prompt = """
You are an expert technical editor. Summarize the provided document in a 
polished, professional, and clean tone. Use markdown headings and bullet 
points where appropriate for readability.
"""

response = qa_chain.invoke(prompt)
print(response['result'])
