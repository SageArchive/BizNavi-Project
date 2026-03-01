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

    all_data = db.get()
    all_docs = all_data.get('documents', [])

    # =================================================================
    # [KEY FIX] Token-based Keyword Match
    # Split the query into individual words (e.g., 'Inbound Price' -> ['inbound', 'price'])
    # =================================================================
    query_words = query_text.lower().strip().split()
    matched_docs = []

    # Check if ALL words in the query exist ANYWHERE in the document
    for doc in all_docs:
        doc_lower = doc.lower()
        if all(word in doc_lower for word in query_words):
            matched_docs.append(doc)

    # Fallback to Vector Search ONLY if keyword match finds nothing
    if not matched_docs:
        print(f"⚠️ No keyword match for '{query_text}'. Falling back to Vector Search...")
        retriever = db.as_retriever(search_kwargs={"k": 5})
        vector_docs = retriever.invoke(query_text)
        matched_docs = [doc.page_content for doc in vector_docs]

    # Format and return top 5 results
    formatted_docs = []
    for i, doc_text in enumerate(matched_docs[:5]):
        formatted_docs.append(f"[Document {i + 1}]\n{doc_text}")

    result_text = "\n\n---\n\n".join(formatted_docs)

    return f"Found relevant policy info:\n\n{result_text}"