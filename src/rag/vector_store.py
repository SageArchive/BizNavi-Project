import os
import pandas as pd
import shutil
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")
FILE_NAME = "Cloud Warehouse Compersion Chart.csv"


def build_vector_db():
    # Reset DB if exists
    if os.path.exists(PERSIST_DIR):
        print(f"Removing old database at {PERSIST_DIR}...")
        shutil.rmtree(PERSIST_DIR)

    # Load data
    filepath = os.path.join(DATA_DIR, FILE_NAME)
    print(f"Loading data from: {filepath}")

    if not os.path.exists(filepath):
        print(f"❌ Error: File not found at {filepath}")
        return
    df = pd.read_csv(filepath)

    # Text Extraction Logic
    documents = []
    for _, row in df.iterrows():
        # Join row values into a single string
        text = " ".join([str(x) for x in row.values if str(x) != 'nan'])
        # Only keep rows with meaningful content length
        if len(text) > 20:
            documents.append(Document(page_content=text, metadata={"source": FILE_NAME}))
    print(f"Extracted {len(documents)} rows of text.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # Embedding & Storage
    print("Embedding and saving to ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create and persist the vector store
    Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    print("✅ Vector DB built successfully.")


if __name__ == "__main__":
    build_vector_db()