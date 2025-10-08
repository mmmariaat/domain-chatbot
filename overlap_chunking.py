from typing import List, Dict, Union, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import read_docs
import re


def chunk_documents(docs: List[Union[str, Dict]], *, chunk_size: int = 1000, chunk_overlap: int = 200, smart: bool = False, per_page: bool = True) -> List[Dict]:
    """
    Chunk a list of documents. Each document can be either:
      - a string (the whole document text), or
      - a dict with at least a 'content' field and optional 'id', 'title', 'category'.

    Returns a list of chunk dicts with keys:
      - id, title, content, category, source_doc
    """
    # Use a splitter that prefers paragraph and sentence boundaries.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    normalized_docs: List[Dict] = []
    for idx, doc in enumerate(docs):
        if isinstance(doc, str):
            normalized_docs.append({
                "id": f"doc_{idx}",
                "title": "",
                "content": doc,
                "category": ""
            })
        elif isinstance(doc, dict):
            # Accept different content keys commonly used
            content = doc.get("content") or doc.get("text") or doc.get("body") or ""
            normalized = {
                "id": str(doc.get("id", f"doc_{idx}")),
                "title": doc.get("title", ""),
                "content": content,
                "category": doc.get("category", "")
            }
            # carry through pages/tables/source metadata if present
            if doc.get("pages"):
                normalized["pages"] = doc.get("pages")
            if doc.get("tables"):
                normalized["tables"] = doc.get("tables")
            if doc.get("source_pdf"):
                normalized["source_pdf"] = doc.get("source_pdf")
            normalized_docs.append(normalized)
        else:
            # Fallback: convert to string
            normalized_docs.append({
                "id": f"doc_{idx}",
                "title": "",
                "content": str(doc),
                "category": ""
            })

    all_chunks: List[Dict] = []
    for doc in normalized_docs:
        # Skip empty content unless there are pages or table docs
        if not doc.get("content") and not doc.get("pages") and not doc.get("tables"):
            continue

        # Smart mode: handle PDFs with pages and table-docs specially
        if smart:
            # If this is a table doc (created by utils), treat it as atomic
            if doc.get("source_pdf") or re.search(r"_table_\d+$", doc.get("id", "")):
                # Do not split tables by default — make them first-class units
                all_chunks.append({
                    "id": f"{doc['id']}_chunk_0",
                    "title": doc.get("title", ""),
                    "content": doc.get("content", ""),
                    "category": doc.get("category", ""),
                    "source_doc": doc.get("source_pdf", doc.get("id")),
                    "page_index": doc.get("page_index")
                })
                continue

            # If doc has pages (PDF), chunk per page to preserve page boundaries
            if per_page and doc.get("pages"):
                for p_idx, page in enumerate(doc.get("pages", [])):
                    page_text = page.get("text", "")
                    if not page_text:
                        continue
                    page_chunks = text_splitter.split_text(page_text)
                    for i, chunk in enumerate(page_chunks):
                        ch = {
                            "id": f"{doc['id']}_p{p_idx}_chunk_{i}",
                            "title": doc.get("title", ""),
                            "content": chunk,
                            "category": doc.get("category", ""),
                            "source_doc": doc.get("id"),
                            "page_index": p_idx
                        }
                        # attach page-level tables if present on this page
                        if page.get("tables"):
                            ch["page_tables"] = page.get("tables")
                        all_chunks.append(ch)
                continue

            # Fallback for smart mode: split the doc content as usual
            content_to_split = doc.get("content", "")
            chunks = text_splitter.split_text(content_to_split)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "id": f"{doc['id']}_chunk_{i}",
                    "title": doc.get("title", ""),
                    "content": chunk,
                    "category": doc.get("category", ""),
                    "source_doc": doc.get("id")
                })
            continue

        # Simple mode (backwards compatible): chunk whole content
        chunks = text_splitter.split_text(doc.get("content", ""))
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{doc['id']}_chunk_{i}",
                "title": doc.get("title", ""),
                "content": chunk,
                "category": doc.get("category", ""),
                "source_doc": doc["id"]
            })
            # attach structural metadata separately if present
            if doc.get("pages") or doc.get("tables"):
                all_chunks[-1]["source_pages"] = doc.get("pages")
                all_chunks[-1]["source_tables"] = doc.get("tables")

    return all_chunks


def load_and_chunk_documents(docs: Optional[List[Union[str, Dict]]] = None, **kwargs) -> List[Dict]:
    """
    Convenience wrapper that reads documents (via `read_docs`) if `docs` is None,
    otherwise chunk the provided list.
    """
    # Allow caller to request structured documents via kwargs (structured=True)
    structured = kwargs.pop("structured", False)
    pdf_parse = kwargs.pop("pdf_parse", False)

    if docs is None:
        docs, paths = read_docs(structured=structured, pdf_parse=pdf_parse)

    # If structured documents were returned as dicts, pass them through
    return chunk_documents(docs, **kwargs)
