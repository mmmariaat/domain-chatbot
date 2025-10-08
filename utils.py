"""
Common utilities for RAG search methods
"""

import os
import glob
from PyPDF2 import PdfReader
import json
try:
    import yaml
except Exception:
    yaml = None


def _parse_structured_file(file_path: str):
    """Parse a structured file (.json, .jsonl, .yml/.yaml) into a list of document dicts.

    Each document dict should have at least a 'content' field. We will also accept
    'id', 'title', and 'category' when present.
    """
    docs = []
    if file_path.endswith(".jsonl") or file_path.endswith(".ndjson"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    docs.append(obj)
                except Exception as e:
                    print(f"Warning: Could not parse line in {file_path}: {e}")
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                obj = json.load(f)
                # Accept either a list of docs or a single doc
                if isinstance(obj, list):
                    docs.extend(obj)
                elif isinstance(obj, dict):
                    # If the dict looks like {id:..., content:...} treat as single doc
                    docs.append(obj)
            except Exception as e:
                print(f"Warning: Could not parse JSON {file_path}: {e}")
    elif (file_path.endswith(".yml") or file_path.endswith(".yaml")) and yaml is not None:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                obj = yaml.safe_load(f)
                if isinstance(obj, list):
                    docs.extend(obj)
                elif isinstance(obj, dict):
                    docs.append(obj)
            except Exception as e:
                print(f"Warning: Could not parse YAML {file_path}: {e}")
    return docs


def _table_to_markdown(table):
    """Convert a 2D list (rows) to a simple markdown table string."""
    if not table:
        return ""
    # Ensure all rows are lists of strings
    rows = [[str(cell) if cell is not None else "" for cell in row] for row in table]
    # Build header (use first row as header if more than one row)
    md = []
    header = rows[0]
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows[1:]:
        md.append("| " + " | ".join(r) + " |")
    return "\n".join(md)


def read_docs(structured: bool = False, pdf_parse: bool = False):
    """Read documents from the `catalog/` directory.

    If `structured` is False (default) returns (docs_texts, paths) where docs_texts
    is a list of strings (document contents) and paths the matching file paths.

    If `structured` is True the function will also load .json/.jsonl/.yml/.yaml
    files and return (docs, paths) where docs is a list of dicts containing at
    least 'content' and optional 'id', 'title', 'category'. For backwards
    compatibility, PDF/TXT/MD files still return plain string documents.
    """
    docs = []
    doc_paths = []

    possible_paths = [
        "catalog/**/*.txt",  # all txt files in catalog and subdirs
        "catalog/**/*.md",   # all markdown files in catalog and subdirs
        "catalog/**/*.pdf",  # all pdf files in catalog and subdirs
    ]

    # If structured reading is enabled, also look for JSON/JSONL/YAML files
    if structured:
        possible_paths = [
            "catalog/**/*.jsonl",
            "catalog/**/*.ndjson",
            "catalog/**/*.json",
            "catalog/**/*.yml",
            "catalog/**/*.yaml",
        ] + possible_paths

    files = []
    for pattern in possible_paths:
        files = glob.glob(pattern, recursive=True)
        if files:
            break

    for file_path in files:
        try:
            if structured and (file_path.endswith('.json') or file_path.endswith('.jsonl') or file_path.endswith('.ndjson') or file_path.endswith('.yml') or file_path.endswith('.yaml')):
                parsed = _parse_structured_file(file_path)
                for obj in parsed:
                    # Normalize to expected dict shape
                    content = obj.get('content') or obj.get('text') or obj.get('body') or ''
                    if not content:
                        # If there's no explicit content, try to stringify the object
                        content = json.dumps(obj)
                    doc = {
                        'id': str(obj.get('id', os.path.basename(file_path))),
                        'title': obj.get('title', ''),
                        'content': content,
                        'category': obj.get('category', '')
                    }
                    docs.append(doc)
                    doc_paths.append(file_path)
            else:
                # For non-structured files (pdf/txt/md) we either return plain strings
                # (backwards compatibility) or normalized dicts when structured=True
                if file_path.endswith(".pdf"):
                    # Use PyPDF2 for basic text extraction, but if pdf_parse=True and
                    # pdfplumber is available, use it to extract tables and page-level text
                    content = ""
                    pages = []
                    tables = []

                    # First try PyPDF2 to get a full-text fallback
                    try:
                        reader = PdfReader(file_path)
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            pages.append({"text": page_text})
                            content += page_text + "\n"
                    except Exception:
                        # If PyPDF2 fails, leave pages/content empty and continue
                        pass

                    content = content.strip()

                    # If pdf_parse requested, try importing pdfplumber at runtime
                    if pdf_parse:
                        try:
                            import importlib
                            _pdfplumber = importlib.import_module('pdfplumber')
                        except Exception:
                            _pdfplumber = None
                        if _pdfplumber is not None:
                            try:
                                with _pdfplumber.open(file_path) as pdf:
                                    pages = []
                                    for p in pdf.pages:
                                        text = p.extract_text() or ""
                                        page_tables = []
                                        try:
                                            for tbl in p.extract_tables():
                                                # Each table is a list of rows (lists of cell strings)
                                                page_tables.append(tbl)
                                                tables.append(tbl)
                                        except Exception:
                                            # extracting tables can fail on some pages, continue
                                            pass
                                        pages.append({"text": text, "tables": page_tables})
                                    # Rebuild a concatenated content from pages if empty
                                    if not content:
                                        content = "\n".join([p.get("text", "") for p in pages]).strip()
                            except Exception as e:
                                print(f"Warning: pdfplumber failed for {file_path}: {e}")
                    # If there's no content at this point, skip
                    if not content:
                        continue

                    if structured:
                        # Try to extract title from PDF metadata (PyPDF2 reader may exist)
                        title = ""
                        try:
                            meta = getattr(reader, 'metadata', None) or getattr(reader, 'documentInfo', None)
                            if meta:
                                if isinstance(meta, dict):
                                    title = meta.get('/Title') or meta.get('title') or ''
                                else:
                                    title = getattr(meta, 'title', '') or ''
                        except Exception:
                            title = ''
                        doc = {
                            'id': os.path.splitext(os.path.basename(file_path))[0],
                            'title': title,
                            'content': content,
                            'category': os.path.basename(os.path.dirname(file_path)),
                            'pages': pages,
                            'tables': tables
                        }
                        docs.append(doc)
                        doc_paths.append(file_path)
                        # If there are tables extracted, also create separate docs for each
                        if tables:
                            for ti, tbl in enumerate(tables):
                                tbl_md = _table_to_markdown(tbl)
                                table_doc = {
                                    'id': f"{doc['id']}_table_{ti}",
                                    'title': f"{doc.get('title','')}' table {ti}",
                                    'content': tbl_md,
                                    'category': os.path.basename(os.path.dirname(file_path)),
                                    'source_pdf': doc['id'],
                                    'page_index': None
                                }
                                docs.append(table_doc)
                                doc_paths.append(file_path)
                    else:
                        docs.append(content)
                        doc_paths.append(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if not content:
                            continue
                        if structured:
                            # Extract title as first non-empty line for txt/md
                            title = ""
                            for line in content.splitlines():
                                line = line.strip()
                                if line:
                                    # If markdown heading, strip leading #'s
                                    if line.startswith('#'):
                                        title = line.lstrip('#').strip()
                                    else:
                                        title = line
                                    break
                            doc = {
                                'id': os.path.splitext(os.path.basename(file_path))[0],
                                'title': title,
                                'content': content,
                                'category': os.path.basename(os.path.dirname(file_path))
                            }
                            docs.append(doc)
                            doc_paths.append(file_path)
                        else:
                            docs.append(content)
                            doc_paths.append(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return docs, doc_paths

def get_doc_info():
    """Get document information for display"""
    # Default to structured-aware call so structured docs are recognized
    docs, paths = read_docs(structured=True)
    
    print(f"📚 Loaded {len(docs)} documents")
    print("\nDocuments:")
    for i, (doc, path) in enumerate(zip(docs, paths)):
        # Get relative path for cleaner display
        rel_path = path.replace("/home/lab-user/techcorp-docs/", "")
        # If the doc is structured dict, show title and preview of content
        if isinstance(doc, dict):
            preview = doc.get('title') or (doc.get('content', '')[:80])
            more = '...' if len(doc.get('content', '')) > 80 else ''
            print(f"{i+1}. [{rel_path}] {preview}{more}")
        else:
            print(f"{i+1}. [{rel_path}] {doc[:80]}{'...' if len(doc) > 80 else ''}")
    
    return docs, paths

