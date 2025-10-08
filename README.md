# 🎯 domain-chatbot
Retrieval-augmented chatbot optimized for structured and contextual domain understanding.

## 💡 How It Works
The chatbot follows a standard Retrieval-Augmented Generation (RAG) pipeline:
1. **Document Loading** – reads and parses course catalog files.  
2. **Chunking** – splits text into overlapping segments for better search.  
3. **Vector Storage** – embeds and stores chunks in a ChromaDB collection.  
4. **Query Processing** – encodes user queries and searches for the most relevant chunks.  
5. **Response Generation** – builds a contextual prompt and sends it to an LLM for the final answer.



## 🚀 Features
  • **Accurate course answers** – delivers precise responses using your uploaded catalogs.
  
  • **RAG-powered retrieval** – ensures grounded and reliable information.
  
  • **Contextual memory** – keeps track of previous questions for natural conversation flow.
  
  • **Flexible model support** – works seamlessly with local or remote LLMs.
  
  • **Minimal Gradio interface** – fast, lightweight, and easy to use.
  

## 💻 Run the Project
**1️⃣ Activate virtual environment**
```bash
source venv/bin/activate

# for Windows:
.\venv\Scripts\activate
```
**2️⃣ Launch the chatbot**
```bash
python chatbot.py

# Or, if you’re using the Gradio interface:
python gradio_app.py  
```

**3️⃣ 🖥️ Launch desktop app (Tauri)**
```bash
cargo tauri dev  
```
## 🧠 Tech Stack

•  **Python 3.10+** – core logic and RAG pipeline

•  **LangChain** – retrieval, chaining, and LLM integration

•  **ChromaDB** – vector database for document storage and search

•  **Gradio** – web-based chat interface

•  **Tauri + Rust** – desktop app wrapper

•  **Ollama / OpenRouter** – local LLM backend for generating responses

•  **SpaCy / Sentence Transformers** – for text embedding and chunking

## 🧩 To Be Improved

• **Structured data parsing** – fix issues with reading and understanding tables or mixed-format PDFs.

• **Chunking optimization** – improve how long and structured texts are split for better retrieval.

• **Prompt flexibility** – make the model handle more general or creative questions smoothly.

• **Document upload tool** – add a built-in option in the UI to upload and process new files easily.
