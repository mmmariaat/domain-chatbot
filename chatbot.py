'''
we are triyng to build a chatbot, that can answer questions based on the context of a given domain.
this chatbot will be a simple AI model, so we need to RAG model, langchain, chroma db, LLM. 

'''
import os
from sentence_transformers import SentenceTransformer
from overlap_chunking import load_and_chunk_documents
from init_chroma import setup_vector_database, save_vector_database_to_file
from vector_search import search_vector_database, augment_prompt_with_context
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

def process_user_query(query: str):
    model = SentenceTransformer('all-MiniLM-L6-v2')  

    # Preprocess query
    cleaned_query = query.lower().strip()
   
    # Convert query to embedding
    query_embedding = model.encode([cleaned_query])
    return model, query_embedding[0]

def generate_response(augmented_prompt: str) -> str:
    """
    Generate response using LLM - OpenRouter.
    """
    model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",  # free model on OpenRouter
    temperature=1.1,
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE") )
    
    response = ChatPromptTemplate.from_template(augmented_prompt)
    chain = response | model

    # invoke and return textual content
    result = chain.invoke({})
    try:
        return result.content
    except Exception:
        # fallback: return str(result)
        return str(result)

# Global in-memory chat history
chat_history = InMemoryChatMessageHistory()


def serialize_history(history: InMemoryChatMessageHistory, max_chars: int = 2000) -> str:
    """Serialize chat history to a single text block suitable for injecting into prompts.

    Keeps the most recent content that fits within max_chars.
    """
    # messages is a list of BaseMessage objects that have .__class__.__name__ and .content
    parts = []
    for msg in history.messages:
        role = msg.__class__.__name__
        content = getattr(msg, 'content', str(msg))
        parts.append(f"{role}: {content}")

    full = "\n".join(parts)
    # trim to last max_chars characters
    if len(full) > max_chars:
        full = full[-max_chars:]
    return full

def run_complete_rag_pipeline(query: str):
    """
    Run the complete RAG pipeline from start to finish.
    """

    chunks = load_and_chunk_documents()

    collection = setup_vector_database(chunks)

    model, query_embedding = process_user_query(query)

    # Perform vector search
    search_results = search_vector_database(collection, query_embedding)

    # Append user question to history
    chat_history.add_user_message(HumanMessage(content=query))

    history_text = serialize_history(chat_history)

    augmented_prompt = augment_prompt_with_context(
        query, search_results, history_text=history_text, use_history=True
    )

    # Generate an answer
    answer = generate_response(augmented_prompt)

    # Save assistant reply into history
    try:
        chat_history.add_ai_message(AIMessage(content=answer))
    except Exception:
        # If AIMessage wrapper isn't available or fails, fallback to raw string
        chat_history.add_ai_message(answer)

    return answer

if __name__ == "__main__":
    while True:
        q = input("\nQuestion (type 'exit' to quit): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print(run_complete_rag_pipeline(q))
    save_vector_database_to_file()