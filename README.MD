# Enterprise AI Data Factory

A lightweight RAG (Retrieval-Augmented Generation) application built with Streamlit, LangChain, and Groq.

## Setup Instructions

1.  **Clone the repository** or copy the files to your project folder.
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Features
- **PDF Processing**: Loads and splits documents into manageable chunks.
- **Embeddings**: Uses `all-MiniLM-L6-v2` for efficient vector representation.
- **Vector Search**: Utilizes `FAISS` for fast similarity retrieval.
- **Inference**: Powered by Llama 3 (70B) via Groq Cloud.
