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

    # =================================================================
    # [KEY FIX] Hybrid Search: Exact Text Match First
    # Vector embeddings often fail at exact keyword matching for specific terms.
    # Since our dataset is small, we first check for an exact text match.
    # =================================================================

    # 1. Fetch all documents directly from the database
    all_data = db.get()
    all_docs = all_data.get('documents', [])

    query_lower = query_text.lower().strip()
    matched_docs = []

    # 2. Perform exact keyword/string matching
    for doc in all_docs:
        if query_lower in doc.lower():
            matched_docs.append(doc)

    # 3. Fallback to Vector Search ONLY if exact match finds nothing
    if not matched_docs:
        print(f"⚠️ No exact match for '{query_text}'. Falling back to Vector Search...")
        retriever = db.as_retriever(search_kwargs={"k": 5})
        vector_docs = retriever.invoke(query_text)
        matched_docs = [doc.page_content for doc in vector_docs]

    # 4. Format and return top 5 results to prevent LLM confusion
    formatted_docs = []
    for i, doc_text in enumerate(matched_docs[:5]):
        formatted_docs.append(f"[Document {i + 1}]\n{doc_text}")

    result_text = "\n\n---\n\n".join(formatted_docs)

    return f"Found relevant policy info:\n\n{result_text}"