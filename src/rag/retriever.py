import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")


def query_warehouse_policy(query_text: str) -> str:
    """
    Searches the warehouse policy/SOP database for relevant information.
    Useful for questions about rules, KPIs, packaging, or inbound/outbound processes.
    """
    if not os.path.exists(PERSIST_DIR):
        return "Error: Vector DB not found. Please run vector_store.py first."

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Retrieve top 3 chunks
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query_text)

    # Combine results
    result_text = "\n\n".join([f"- {doc.page_content}" for doc in docs])
    return f"Found relevant policy info:\n{result_text}"