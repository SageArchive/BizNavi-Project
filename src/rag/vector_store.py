import os
import pandas as pd
import shutil
import re
from dotenv import load_dotenv

load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
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

    # Text Extraction Logic with Section Tracking
    documents = []

    # Set the default section for the top part of the CSV
    current_section = "General Pricing and Commercials"
    last_topic = ""

    # Helper function to clean text
    def clean_text(text):
        t = text.replace('â‚¹', '₹') \
            .replace('â€¢Â', '') \
            .replace('â€“', '-') \
            .replace('Â', '')
        # Replace multiple spaces with a single space
        t = re.sub(r'\s+', ' ', t).strip()
        # Remove leading bullet points
        if t.startswith("•"): t = t[1:].strip()
        return t

    for _, row in df.iterrows():
        # Explicitly extract topic, description, and value using column index
        topic = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        desc = str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else ""
        val = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""

        # Detect Section Headers (A, B, C, D) and update current_section
        if topic.startswith('(A)') or topic.startswith('(B)') or topic.startswith('©') or topic.startswith('(D)'):
            # Auto-correct typo '© EXCLUSIONS' to '(C) EXCLUSIONS'
            current_section = clean_text(topic.replace('©', '(C)'))
            last_topic = ""  # Reset topic since a new section started
            continue

        # Skip table header rows
        if topic in ["Heads", "Performance Indicators", "Performance Measure"]:
            continue

        # Handle merged cells (forward fill logic)
        if topic:
            last_topic = topic
        else:
            topic = last_topic

        # Execute text cleaning
        clean_topic = clean_text(topic)
        clean_desc = clean_text(desc)
        clean_val = clean_text(val)

        # Only save to DB if there is meaningful data in description or value
        if len(clean_desc) > 1 or len(clean_val) > 1:
            text_parts = [f"Section: {current_section}"]

            # Apply different labeling templates based on the section
            if current_section == "General Pricing and Commercials":
                if clean_topic: text_parts.append(f"Item: {clean_topic}")
                if clean_desc: text_parts.append(f"Shiprocket Price: {clean_desc}")
                if clean_val: text_parts.append(f"INCREFF Price: {clean_val}")
            else:
                if clean_topic: text_parts.append(f"Policy Topic: {clean_topic}")
                if clean_desc: text_parts.append(f"Description: {clean_desc}")
                if clean_val: text_parts.append(f"Value or Limit: {clean_val}")

            text = "\n".join(text_parts)
            documents.append(Document(page_content=text, metadata={"source": FILE_NAME}))

    print(f"Extracted {len(documents)} rows of text.")

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # Embedding & Storage
    print("Embedding and saving to ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Create and persist the vector store
    Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIR)
    print("✅ Vector DB built successfully.")


if __name__ == "__main__":
    build_vector_db()