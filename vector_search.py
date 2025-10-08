from typing import List, Dict, Optional

def search_vector_database(collection, query_embedding, top_k: int = 3):
    """
    Query the vector collection and return a list of results with similarity scores.
    """
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Process and display results
    search_results = []
    for i, (doc_id, distance, content, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['documents'][0],
        results['metadatas'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity
        search_results.append({
            'id': doc_id,
            'content': content,
            'metadata': metadata,
            'similarity': similarity
        })

    return search_results


def augment_prompt_with_context(
    query: str,
    search_results: List[Dict],
    history_text: Optional[str] = None,
    use_history: bool = True,
) -> str:
    """
    Build augmented prompt with retrieved context for LLM. If `use_history` is True and
    `history_text` is provided, the chat history will be injected and given priority over
    the vector search context in case of conflict.
    """
    # Assemble context from search results
    context_parts = []
    for i, result in enumerate(search_results, 1):
        title = result.get('metadata', {}).get('title', f'Doc {i}')
        context_parts.append(f"Source {i}: {title}\n{result['content']}")

    context = "\n\n".join(context_parts) if context_parts else "(no retrieved documents)"

    # Prepare history section if available
    history_section = history_text if (use_history and history_text) else None

    # Build augmented prompt - before sending to LLM
    priority_note = (
        "IMPORTANT: If the chat history contains information that conflicts with the"
        " retrieved catalog context, prefer the chat history and act accordingly."
        if history_section
        else ""
    )

    augmented_prompt = f"""
You are a knowledgeable and friendly university assistant.
{priority_note}
Use the following course catalog information and any relevant reasoning to answer the user's question.

CHAT HISTORY:
{history_section or '(no chat history available)'}

COURSE CATALOG CONTEXT:
{context}

STUDENT QUESTION:
{query}

GUIDELINES:
- dont give answers that are not relevant to the question, make sure your answers are based on the catalog context and chat history.
- Never mention the source of the information in the answer.
- dont say hello, hi, or any other pleasantries unless the user does first.
- rely on the catalog context and the chat history (history has priority if enabled) for your answers.
- Be clear, helpful, and concise, dont give answers that are not relevant to the question.
- When possible, include course names, prerequisites, and details.
- If the answer cannot be found in the catalog or history, say so naturally.
"""

    return augmented_prompt