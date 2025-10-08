import chromadb
import json, os
from sentence_transformers import SentenceTransformer
from utils import read_docs
from typing import List, Dict

model = SentenceTransformer('all-MiniLM-L6-v2')


def setup_vector_database(chunks: List[Dict]):
    """
    Set up ChromaDB vector database and store document chunks.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db") # this makes the db persistent
    try:
        collection = chroma_client.create_collection(
            name="catalog",  
            metadata={"hnsw:space": "cosine"}  # using cosine similarity
        )
    except Exception:
        # Collection already exists, get it
        collection = chroma_client.get_collection("catalog")
   
    # Prepare data for storage with embeddings and richer metadata
    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk.get("content", "") for chunk in chunks]
    # Build metadata with optional fields
    metadatas = []
    for chunk in chunks:
        meta = {
            "title": chunk.get("title", ""),
            "category": chunk.get("category", ""),
            "source": chunk.get("source_doc") or chunk.get("source_pdf") or "",
        }
        # Add page-related metadata if present
        if "page_index" in chunk:
            meta["page_index"] = chunk["page_index"]
        if "page_tables" in chunk:
            # boolean flag indicating whether this chunk's page contained tables
            meta["has_page_tables"] = bool(chunk.get("page_tables"))
        metadatas.append(meta)

    # Compute embeddings for all documents using the sentence-transformers model
    # The model.encode accepts a list of texts and returns a 2D array
    embeddings = model.encode(documents, show_progress_bar=False)

    # If collection is empty, add everything. Otherwise filter out existing ids.
    try:
        existing = collection.get(ids=ids)
        existing_ids = set(existing.get("ids", []))
    except Exception:
        existing_ids = set()

    to_add = [i for i, cid in enumerate(ids) if cid not in existing_ids]
    if to_add:
        add_ids = [ids[i] for i in to_add]
        add_docs = [documents[i] for i in to_add]
        add_metas = [metadatas[i] for i in to_add]
        add_embs = [embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else embeddings[i] for i in to_add]
        collection.add(ids=add_ids, documents=add_docs, metadatas=add_metas, embeddings=add_embs)
    return collection

def save_vector_database_to_file( filename="vectordb_backup.json"):
    """
    Save the vector database collection to a JSON file for persistence.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection("catalog")

    with open("vectordb_backup.json", "w") as f:
        json.dump(
            {
                "documents": collection.get()["documents"],
                "ids": collection.get()["ids"],
                "metadatas": collection.get()["metadatas"],
                "count": collection.count()
            },
            f,
            indent=2
        )
